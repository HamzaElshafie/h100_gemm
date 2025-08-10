#pragma once

#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>

template <const uint TILE_SIZE_M, const uint TILE_SIZE_N, const uint TILE_SIZE_K,
        const uint WARP_TILE_M, const uint WARP_TILE_K, const uint WARP_STEPS_K,
        const uint ROWS_PER_THREAD, const uint COLS_PER_THREAD, const uint NUM_THREADS,
        bool PAD_SMEM_A = false, bool PAD_SMEM_B = false>
__global__ void __launch_bounds__(NUM_THREADS) 
    gemm_warptiling_bf16(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C,
        int M, int N, int K, float alpha, float beta) {

        static_assert(TILE_SIZE_K % WARP_TILE_K == 0, "TILE_SIZE_K must tile by WARP_TILE_K");

        // Allocate shared memory
        __shared__ __nv_bfloat16 sharedA[TILE_SIZE_M * TILE_SIZE_N];
        __shared__ __nv_bfloat16 sharedB[TILE_SIZE_N * TILE_SIZE_K];

        // Identify the tile of C this thread block is responsible for
        const uint block_row = blockIdx.y;
        const uint block_column = blockIdx.x;

        constexpr uint WARP_STEPS_M = (WARP_TILE_M * WARP_TILE_K) / (WARPSIZE * (ROWS_PER_THREAD * COLS_PER_THREAD)); // 1
        // Warp subtile is WARP_SUB_M x WARP_SUB_K
        constexpr uint WARP_SUB_M = WARP_TILE_M / WARP_STEPS_M;
        constexpr uint WARP_SUB_K = WARP_TILE_K / WARP_STEPS_K;

        // Identify the warp tile position
        const uint warp_idx = threadIdx.x / WARPSIZE; // threads 0,..,31 --> warp_idx = 0,  threads 32,..,63 --> warp_idx = 1 etc
        const uint warps_per_row = TILE_SIZE_K / WARP_TILE_K; // 2
        const uint warp_row = warp_idx / warps_per_row; // (0, 1). Given our launch configs ofc
        const uint warp_col = warp_idx % warps_per_row; // (0, 1)

        // Identify the thread position within the warp tile
        uint lane = threadIdx.x % WARPSIZE;
        const uint threads_per_subtile_row = WARP_SUB_K / COLS_PER_THREAD; // 16/4 = 4 threads per row
        const uint thread_row_in_sub = lane / threads_per_subtile_row; // Eg. 6/4 = 1.5 = 1. Row 1
        const uint thread_col_in_sub = lane % threads_per_subtile_row; // Eg. 5%4 = 1. Col 1

        const uint ty = thread_row_in_sub;
        const uint tx = thread_col_in_sub;

        // Move pointers from A[0], B[0] and C[0] to the starting positions of the tile
        A += block_row * TILE_SIZE_M * N;                                  // Move pointer (block_row * TILE_SIZE_M) rows down
        B += block_column * TILE_SIZE_K;                                   // Move pointer (block_column * TILE_SIZE_K) columns to the right
        C += (block_row * TILE_SIZE_M * K) + (block_column * TILE_SIZE_K); // Move pointer (block_row * TILE_SIZE_M * K) rows down then (block_column * TILE_SIZE_K) columns to the right
        
        const uint VEC_CHUNKS_N = TILE_SIZE_N / 4;
        const uint VEC_CHUNKS_K = TILE_SIZE_K / 4;

        // Map each thread to one 4-float chunk that it will load. We will have to offset as we have fewer threads than elements to cover (offset by 32)
        const uint smem_ty_A = threadIdx.x / VEC_CHUNKS_N; // --> 0, ..., 31
        const uint smem_tx_A = threadIdx.x % VEC_CHUNKS_N; // --> 0, 1, 2, 3
        // If we give each thread a vector load of 4 elements along TILE_SIZE_N, how many different rows of sharedA can we cover in one pass through all the threads?
        const uint strideA = (NUM_THREADS * 4) / TILE_SIZE_N; // 32 rows per pass

        const uint smem_ty_B = threadIdx.x / TILE_SIZE_K; // --> 0, 1, 2, 3
        const uint smem_tx_B = threadIdx.x % TILE_SIZE_K; // --> 0, ...., 31
        // If we give each thread a vector load of 4 elements along TILE_SIZE_K, how many different rows of sharedB can we cover in one pass through all the threads?
        const uint strideB = (NUM_THREADS * 4) / TILE_SIZE_K; // 4 rows per pass
    }