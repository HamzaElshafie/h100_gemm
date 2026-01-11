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

template <const uint TILE_SIZE_M, const uint TILE_SIZE_K, const uint TILE_SIZE_N,
          const uint WGMMA_M, const uint WGMMA_K, const uint WGMMA_N, const uint NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
gemm_bf16_wgmma_tma_shapes(const CUtensorMap* tensorMapA, const CUtensorMap* tensorMapB, bf16* C,
    int M, int K, int N, float alpha, float beta) {
        // assert WGMMA_N = TILE_SIZE_N
        static_assert(WGMMA_N == TILE_SIZE_N, "WGMMA_N must be == TILE_SIZE_N");

        // Allocate SMEM 
        __shared__ alignas(128) bf16 sharedA[TILE_SIZE_M * TILE_SIZE_K];
        __shared__ alignas(128) bf16 sharedB[TILE_SIZE_K * TILE_SIZE_N];

        // Init each thread's register
        float d[TILE_SIZE_M / WGMMA_M][WGMMA_N / WGMMA_K][8]; // Where WGMMA_K will always be 16
        memset(d, 0, sizeof(d));

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
                tokenA = cuda::device::barrier_arrive_tx(barA, 1, sizeof(sharedA));
                cde::cp_async_bulk_tensor_2d_global_to_shared(&sharedB[0], tensorMapB, block_k_iter * TILE_SIZE_K, num_block_n * TILE_SIZE_N, barB);
                tokenB = cuda::device::barrier_arrive_tx(barB, 1, sizeof(sharedB));
            } else {
                // Other threads just arrive.
                tokenA = barA.arrive();
                tokenB = barB.arrive();
            }
            // All threads wait for async loads to complete
            barA.wait(std::move(tokenA));
            barB.wait(std::move(tokenB));
            __syncthreads();

            // 2. Compute phase
            // TODO
        }

        // 3. Epilogue store phase
        // TODO
    }