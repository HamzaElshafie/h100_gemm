#pragma once

#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include "utils.h"
#include "hopper_wgmma_utils.cuh"
#include "hopper_tma_utils.h"

// Alias for simplicity
using bf16 = __nv_bfloat16;
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

template <int TILE_SIZE_M, int TILE_SIZE_K, int TILE_SIZE_N>
struct SMem {
    alignas(128) bf16 A[TILE_SIZE_M * TILE_SIZE_K];
    alignas(128) bf16 B[TILE_SIZE_K * TILE_SIZE_N];
};

template <const uint TILE_SIZE_M, const uint TILE_SIZE_K, const uint TILE_SIZE_N,
          const uint WGMMA_M, const uint WGMMA_K, const uint WGMMA_N, const uint NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
gemm_bf16_wgmma_tma_shapes(const CUtensorMap* tensorMapA, const CUtensorMap* tensorMapB, bf16* C,
    int M, int K, int N, float alpha, float beta) {
        // assert WGMMA_N = TILE_SIZE_N
        static_assert(WGMMA_N == TILE_SIZE_N, "WGMMA_N must be == TILE_SIZE_N");
        static_assert(NUM_THREADS % 128 == 0, "NUM_THREADS must be divisible by warp group size (128)");
        static_assert(TILE_SIZE_K % WGMMA_K == 0, "TILE_SIZE_K must be divisible by WGMMA_K");

        // Allocate SMEM 
        extern __shared__ SMem<TILE_SIZE_M, TILE_SIZE_K, TILE_SIZE_N> s;
        bf16* sharedA = s.A;
        bf16* sharedB = s.B;

        // Init each thread's register
        float d[TILE_SIZE_M / WGMMA_M][WGMMA_N / WGMMA_K][8]; // Where WGMMA_K will always be 16
        memset(d, 0, sizeof(d));

        // How many warp groups are in this block (one warp group = 128 threads)
        constexpr int num_warp_groups = NUM_THREADS / 128;

        // How many M rows of the output tile each warp group is responsible for
        // @example: TILE_SIZE_M = 128, NUM_THREADS = 256 -> num_warp_groups = 2 -> 64 rows per warp group
        constexpr int rows_per_warp_group = TILE_SIZE_M / num_warp_groups;

        int warp_group_idx = threadIdx.x / 128;

        const int num_blocks_k = CEIL_DIV(K, TILE_SIZE_K);
        int num_block_n = blockIdx.x % CEIL_DIV(N, TILE_SIZE_N);
        int num_block_m = blockIdx.x / CEIL_DIV(N, TILE_SIZE_N);

        // set SMEM barriers for A and B
        #pragma nv_diag_suppress static_var_with_dynamic_init
        __shared__ barrier barA;
        #pragma nv_diag_suppress static_var_with_dynamic_init
        __shared__ barrier barB;

        if (threadIdx.x == 0) {
            // A single thread initializes the total expected arrival count.
            // barrier expects blockDim.x (=N) arrivals before it is released. This is the countdown counter the
            // async barrier tracks.
            // @cite https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html#a-barrier-s-phase-arrival-countdown-completion-and-reset
            init(&barA, blockDim.x);
            init(&barB, blockDim.x);
            cde::fence_proxy_async_shared_cta();
        }
        __syncthreads();

        // 1. Loading phase (TMA)
        // Set up two tokens for tracking thread's arrivals
        barrier::arrival_token tokenA;
        barrier::arrival_token tokenB;

        // Outer loop across the shared dim
        for (int block_k_iter = 0; block_k_iter < num_blocks_k; block_k_iter++) {
            if (threadIdx.x == 0) {
                // Initiate bulk tensor copy.
                cde::cp_async_bulk_tensor_2d_global_to_shared(&sharedA[0], tensorMapA, block_k_iter * TILE_SIZE_K, num_block_m * TILE_SIZE_M, barA);
                // Arrive on the barrier and tell how many bytes are expected to come in.
                tokenA = cuda::device::barrier_arrive_tx(barA, 1, sizeof(s.A));
                cde::cp_async_bulk_tensor_2d_global_to_shared(&sharedB[0], tensorMapB, block_k_iter * TILE_SIZE_K, num_block_n * TILE_SIZE_N, barB);
                tokenB = cuda::device::barrier_arrive_tx(barB, 1, sizeof(s.B));
            } else {
                // Other threads just arrive.
                tokenA = barA.arrive();
                tokenB = barB.arrive();
            }
            // All threads wait for async loads to complete
            barA.wait(std::move(tokenA));
            barB.wait(std::move(tokenB));
            __syncthreads();

            // 2. Compute phase using WGMMA tensor cores instructions
            warpgroup_arrive();
            // Outer loop over TILE_SIZE_M in WGMMA_M steps
            // If we have two warp groups, we let each work on a different partition of TILE_SIZE_M
            // @example: 
            #pragma unroll
            for (int m_iter = 0; m_iter < rows_per_warp_group / WGMMA_M; m_iter++) {
                bf16* sharedA_wgmma_tile_base = sharedA + ((warp_group_idx * rows_per_warp_group) + (m_iter * WGMMA_M)) * TILE_SIZE_K;
                // Inner loop iterating over TILE_SIZE_K in WGMMA_K steps
                #pragma unroll
                for (int k_iter = 0; k_iter < TILE_SIZE_K / WGMMA_K; k_iter++) {
                    wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[m_iter], &sharedA_wgmma_tile_base[k_iter * WGMMA_K], &sharedB[k_iter * WGMMA_K]);
                }
            }
            warpgroup_commit_batch(); // asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
            warpgroup_wait<0>();      // asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
        }

        // 3. Epilogue store phase
        int tid = threadIdx.x % 128; // “folds” threads so the indexing logic assumes a single 128 thread warp group.
        int lane = tid % 32;
        int warp = tid / 32;
        uint32_t row = warp * 16 + lane / 4;
        // @note C is column-major
        bf16* block_C = C + (num_block_n * TILE_SIZE_N * M) + (num_block_m * TILE_SIZE_M);
        for (int m_iter = 0; m_iter < rows_per_warp_group / WGMMA_M; m_iter++) {
            int row_tile_base_C = C + (warp_group_idx * rows_per_warp_group) + (m_iter * WGMMA_M) * M;
        }
    }