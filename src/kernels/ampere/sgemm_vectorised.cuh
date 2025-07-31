#pragma once

#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>


template <const uint TILE_SIZE_M, const uint TILE_SIZE_N, const uint TILE_SIZE_K,  const uint ROWS_PER_THREAD, const uint COLS_PER_THREAD>
__global__ void sgemm_vectorised(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, float alpha, float beta) {
    // Allocate shared memory
    __shared__ float sharedA[TILE_SIZE_M * TILE_SIZE_N];
    __shared__ float sharedB[TILE_SIZE_N * TILE_SIZE_K];

    // Identify the tile of C this thread block is responsible for
    const uint block_row = blockIdx.y;
    const uint block_column = blockIdx.x;

    // Calculate position of thread within tile (Remapping from 1-D to 2-D) Note --> Each thread is a grid in itself hanlding ROWS_PER_THREAD x COLS_PER_THREAD
    const uint ty = threadIdx.x / (TILE_SIZE_K / COLS_PER_THREAD);
    const uint tx = threadIdx.x % (TILE_SIZE_K / COLS_PER_THREAD);

    // Move pointers from A[0], B[0] and C[0] to the starting positions of the tile
    A += block_row * TILE_SIZE_M * N;                                  // Move pointer (block_row * TILE_SIZE_M) rows down
    B += block_column * TILE_SIZE_K;                                   // Move pointer (block_column * TILE_SIZE_K) columns to the right
    C += (block_row * TILE_SIZE_M * K) + (block_column * TILE_SIZE_K); // Move pointer (block_row * TILE_SIZE_M * K) rows down then (block_column * TILE_SIZE_K) columns to the right

    // Map each thread to one 4-float chunk that it will load.
    const uint smem_ty_A = threadIdx.x / (TILE_SIZE_N / 4); // --> 0, ..., 127
    const uint smem_tx_A = threadIdx.x % (TILE_SIZE_N / 4); // --> 0, 1

    const uint smem_ty_B = threadIdx.x / (TILE_SIZE_K / 4); // --> 0, ..., 7
    const uint smem_tx_B = threadIdx.x % (TILE_SIZE_K / 4); // --> 0, ..., 31

    // Calculate how many tiles we have
    const uint num_tiles = CEIL_DIV(N, TILE_SIZE_N);
    float thread_results[ROWS_PER_THREAD * COLS_PER_THREAD] = {0.0f};
    float reg_m[ROWS_PER_THREAD] = {0.0f};
    float reg_k[COLS_PER_THREAD] = {0.0f};

    // Outer loop iterate over tiles
    for (int t = 0; t < num_tiles; t++) {
        // Populate smem using vector loads
        float4 tempA = reinterpret_cast<const float4*>(&A[smem_ty_A * N + smem_tx_A*4])[0]; // [0] dereference issues one ld.global.nc.v4.f32 
        // Transpose A (instead of 128x8 previously for ex, now it will be 8x128)
        sharedA[(smem_tx_A * 4 + 0) * TILE_SIZE_M + smem_ty_A] = tempA.x;
        sharedA[(smem_tx_A * 4 + 1) * TILE_SIZE_M + smem_ty_A] = tempA.y;
        sharedA[(smem_tx_A * 4 + 2) * TILE_SIZE_M + smem_ty_A] = tempA.z;
        sharedA[(smem_tx_A * 4 + 3) * TILE_SIZE_M + smem_ty_A] = tempA.w;

        float4 tempB = reinterpret_cast<const float4*>(&B[smem_ty_B * K + smem_tx_B*4])[0];
        reinterpret_cast<float4*>(&sharedB[smem_ty_B * TILE_SIZE_K + smem_tx_B*4])[0] = tempB;

        __syncthreads();

        // Outer loop over shared dimension N
        for (int i = 0; i < TILE_SIZE_N; i++) {
            // Load into registers one "col" (its acc row now) from sharedA and one row from sharedB
            for (int row = 0; row < ROWS_PER_THREAD; row++) {
                uint global_smem_row_idx = ty * ROWS_PER_THREAD + row;
                reg_m[row] = sharedA[i * TILE_SIZE_M + global_smem_row_idx]; // i will be the same for the whole "column" although since its transposed we are
                                                                            // accessing same row. Notice how we also skip rows by TILE_SIZE_M now
            }
            for (int col = 0; col < COLS_PER_THREAD; col++) {
                uint global_smem_col_idx = tx * COLS_PER_THREAD + col;
                reg_k[col] = sharedB[i * TILE_SIZE_K + global_smem_col_idx];
            }

            // Calculate outer product between reg_m and reg_k to produce the partial results matrix of the thread 
            for (uint m = 0; m < ROWS_PER_THREAD; m++) {
                for (uint k = 0; k < COLS_PER_THREAD; k++) {
                    thread_results[m * COLS_PER_THREAD + k] += reg_m[m] * reg_k[k]; // --> (ROWS_PER_THREAD x COLS_PER_THREAD) matrix
                }
            }
        }
        __syncthreads();

        A += TILE_SIZE_N; // Move right
        B += TILE_SIZE_N * K; // Move down                               
    }
    // Write results of the thread back to C
    for (uint row = 0; row < ROWS_PER_THREAD; row++) {
        // handle COLS_PER_THREAD in chunks of 4
        for (uint col = 0; col < COLS_PER_THREAD; col+=4) {
            uint global_row_idx = ty * ROWS_PER_THREAD + row;
            uint global_col_idx = tx * COLS_PER_THREAD + col;
            float4 tempC = reinterpret_cast<float4*>(&C[global_row_idx * K + global_col_idx])[0];
            tempC.x = (alpha * thread_results[row * COLS_PER_THREAD + col]) + (beta * tempC.x);
            tempC.y = (alpha * thread_results[row * COLS_PER_THREAD + col+1]) + (beta * tempC.y);
            tempC.z = (alpha * thread_results[row * COLS_PER_THREAD + col+2]) + (beta * tempC.z);
            tempC.w = (alpha * thread_results[row * COLS_PER_THREAD + col+3]) + (beta * tempC.w);

            reinterpret_cast<float4*>(&C[global_row_idx * K + global_col_idx])[0] = tempC;
        }
    }
}
