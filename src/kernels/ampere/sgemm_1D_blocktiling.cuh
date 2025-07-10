#pragma once

#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>

template <const uint TILE_SIZE_M, const uint TILE_SIZE_N, const uint TILE_SIZE_K,  const uint ROWS_PER_THREAD>
__global__ void sgemm_1D_blocktiling(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, float alpha, float beta) {
         // Allocate shared memory
        __shared__ float sharedA[TILE_SIZE_M * TILE_SIZE_N];
        __shared__ float sharedB[TILE_SIZE_N * TILE_SIZE_K];

        // Identify the tile of C this thread block is responsible for 
        const uint block_row = blockIdx.y;
        const uint block_column = blockIdx.x;

        // Calculate position of thread within tile (Remapping from 1-D to 2-D)
        const uint ty = threadIdx.x / TILE_SIZE_K; // (0, TILE_SIZE_K-1)
        const uint tx = threadIdx.x % TILE_SIZE_K; // (0, TILE_SIZE_K-1)

        // Move pointers from A[0], B[0] and C[0] to the starting positions of the tile
        A += block_row * TILE_SIZE_M * N; // Move pointer (block_row * TILE_SIZE_M) rows down
        B += block_column * TILE_SIZE_K; // Move pointer (block_column * TILE_SIZE_K) columns to the right 
        C += (block_row * TILE_SIZE_M * K) + (block_column * TILE_SIZE_K); // Move pointer (block_row * TILE_SIZE_M * K) rows down then (block_column * TILE_SIZE_K) columns to the right

        // Calculate position of thread within shared memory tile 
        const uint smem_ty_A = threadIdx.x / TILE_SIZE_N;
        const uint smem_tx_A = threadIdx.x % TILE_SIZE_N;

        const uint smem_ty_B = threadIdx.x / TILE_SIZE_K;
        const uint smem_tx_B = threadIdx.x % TILE_SIZE_K;

        // Calculate how many tiles we have
        const uint num_tiles = (K * TILE_SIZE_K + 1) / TILE_SIZE_K;

        // Initialise results array to match how many elements each thread is handing (This will be stores in register file of each thread)
        float thread_results[ROWS_PER_THREAD] = {0.0f};

        // Iterate over tiles (Phase 1: Loading data)
        for (int t = 0; t < num_tiles; t++) {
            sharedA[smem_ty_A * TILE_SIZE_N + smem_tx_A] = A[smem_ty_A * N + smem_tx_A];
            sharedB[smem_ty_B * TILE_SIZE_K + smem_tx_B] = B[smem_ty_B * K + smem_tx_B];

            // Barrier synchronisation until all threads load smem tiles
            __syncthreads();

            // Loop over the shared dimension between the smem tiles, which is TILE_SIZE_N
            for (int i = 0; i < TILE_SIZE_N; i++) {
                float fixed_B = sharedB[i * TILE_SIZE_K + tx]; // Fixate one value from sharedB every iteration
                for (int row = 0; row < ROWS_PER_THREAD; row++) {
                    uint global_row_idx = ty * ROWS_PER_THREAD + row;
                    thread_results[row] += sharedA[global_row_idx * TILE_SIZE_N + i] * fixed_B;
                }
            }
            // Barrier synchronisation until all threads finish computing their results
            __syncthreads();
        }
        // Write results back to C
        for (int row = 0; row < ROWS_PER_THREAD; row++) {
            uint global_row_idx = ty * ROWS_PER_THREAD + row;
            C[global_row_idx * K + tx] = (alpha * thread_results[row]) + (beta * C[global_row_idx * K + tx]);
        }
    }