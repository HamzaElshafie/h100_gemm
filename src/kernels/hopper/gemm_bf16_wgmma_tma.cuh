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

template <const uint TILE_SIZE_M, const uint TILE_SIZE_K, const uint TILE_SIZE_N,
          const uint WGMMA_M, const uint WGMMA_K, const uint WGMMA_N, const uint NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
gemm_bf16_wgmma_tma(const CUtensorMap* __restrict__ tensorMapA, const CUtensorMap* __restrict__ tensorMapB, bf16* __restrict__ C,
    int M, int N, int K, float alpha, float beta) {
    // Allocate SMEM
    __shared__ alignas(128) bf16 sharedA[TILE_SIZE_M * TILE_SIZE_K];
    __shared__ alignas(128) bf16 sharedB[TILE_SIZE_K * TILE_SIZE_N];
    // Initialise thread's accumilator
    // d[4][8] = 32 floats per thread
    float d[WGMMA_N / 16][8];
    memset(d, 0, sizeof(d));

    const int num_blocks_k = CEIL_DIV(K, TILE_SIZE_K);
    int num_block_n = blockIdx.x % CEIL_DIV(N, TILE_SIZE_N);
    int num_block_m = blockIdx.x / CEIL_DIV(N, TILE_SIZE_N);

    // SMEM barriers for A and B
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier barA; 
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier barB;

    if (threadIdx.x == 0) {
        init(&barA, blockDim.x);
        init(&barB, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    barrier::arrival_token tokenA, tokenB;
    for (int block_k_iter = 0; block_k_iter < num_blocks_k; block_k_iter++) {
        // Async loads (Only 1 thread launches the TMA op)
        if (threadIdx.x == 0) {
            // Thread 0 launches async bulk tensor copy operations for both matrices
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sharedA[0], tensorMapA, block_k_iter * TILE_SIZE_K, num_block_m * TILE_SIZE_M, barA);
            // Signal barrier and wait for both loads to complete
            tokenA = cuda::device::barrier_arrive_tx(barA, 1, sizeof(sharedA));
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sharedB[0], tensorMapB, block_k_iter * TILE_SIZE_K, num_block_n * TILE_SIZE_N, barB);
            tokenB = cuda::device::barrier_arrive_tx(barB, 1, sizeof(sharedB));
        }
        else {
            // Other threads arrive at barrier to synchronise data loads
            tokenA = barA.arrive();
            tokenB = barB.arrive();
        }
        // All threads wait for async loads to complete
        barA.wait(std::move(tokenA));
        barB.wait(std::move(tokenB));
        __syncthreads();

        // Compute phase using WGMMA tensor cores
        warpgroup_arrive(); // asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        wgmma64<1, 1, 1, 0, 0>(d, &sharedA[0], &sharedB[0]);
        wgmma64<1, 1, 1, 0, 0>(d, &sharedA[WGMMA_K], &sharedB[WGMMA_K]);
        wgmma64<1, 1, 1, 0, 0>(d, &sharedA[2 * WGMMA_K], &sharedB[2 * WGMMA_K]);
        wgmma64<1, 1, 1, 0, 0>(d, &sharedA[3 * WGMMA_K], &sharedB[3 * WGMMA_K]);
        warpgroup_commit_batch(); // asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        warpgroup_wait<0>();      // asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
    }

    // Store results from accumulator to global memory
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;
    uint32_t row = warp * 16 + lane / 4;
    bf16 *block_C = C + num_block_n * TILE_SIZE_N * M + num_block_m * TILE_SIZE_M;

    for (int m_it = 0; m_it < TILE_SIZE_M / WGMMA_M; ++m_it) {
        for (int n_it = 0; n_it < TILE_SIZE_N / WGMMA_N; ++n_it) {
            for (int w = 0; w < WGMMA_N / 16; ++w) { // w = {0, 1, 2, 3}
                // (16 * w) selects the base col of the 16 col block
                int col = 16 * w + 2 * (tid % 4);
                #define IDX(i, j) ((j + n_it * WGMMA_N) * M + ((i) + m_it * WGMMA_M))

                // Apply alpha scaling to accumulator results and add beta*C
                block_C[IDX(row, col)] = __float2bfloat16(alpha * d[w][0] + beta * __bfloat162float(block_C[IDX(row, col)]));
                block_C[IDX(row, col + 1)] = __float2bfloat16(alpha * d[w][1] + beta * __bfloat162float(block_C[IDX(row, col + 1)]));
                block_C[IDX(row + 8, col)] = __float2bfloat16(alpha * d[w][2] + beta * __bfloat162float(block_C[IDX(row + 8, col)]));
                block_C[IDX(row + 8, col + 1)] = __float2bfloat16(alpha * d[w][3] + beta * __bfloat162float(block_C[IDX(row + 8, col + 1)]));

                block_C[IDX(row, col + 8)] = __float2bfloat16(alpha * d[w][4] + beta * __bfloat162float(block_C[IDX(row, col + 8)]));
                block_C[IDX(row, col + 9)] = __float2bfloat16(alpha * d[w][5] + beta * __bfloat162float(block_C[IDX(row, col + 9)]));
                block_C[IDX(row + 8, col + 8)] = __float2bfloat16(alpha * d[w][6] + beta * __bfloat162float(block_C[IDX(row + 8, col + 8)]));
                block_C[IDX(row + 8, col + 9)] = __float2bfloat16(alpha * d[w][7] + beta * __bfloat162float(block_C[IDX(row + 8, col + 9)]));

                #undef IDX
            }
        }
    } 
}