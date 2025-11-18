#pragma once

#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

/**
 * @brief 1D Register Tiling GEMM for BF16 (accumulates in FP32).
 *
 * Computes the matrix multiplication C = alpha * (A @ B) + beta * C,
 * where A is of size (MxN), B is of size (NxK), and C is of size (MxK).
 * This implementation uses 1D block tiling for improved performance where each
 * thread is responsible for computing multiple output elements.
 *
 * @tparam TILE_SIZE_M Number of rows in each tile of matrix A
 * @tparam TILE_SIZE_N Number of columns in each tile of matrix A and rows in each tile of matrix B
 * @tparam TILE_SIZE_K Number of columns in each tile of matrix B
 * @tparam ROWS_PER_THREAD Number of rows each thread computes
 * @param A       Pointer to input matrix A, stored in row-major order
 * @param B       Pointer to input matrix B
 * @param C       Pointer to output matrix C
 * @param M       Number of rows in matrix A and C
 * @param N       Number of columns in A and rows in B (shared dimension)
 * @param K       Number of columns in matrices B and C
 * @param alpha   Scalar multiplier for the matrix product (A @ B)
 * @param beta    Scalar multiplier for the existing values in matrix C
 */
template <const uint TILE_SIZE_M, const uint TILE_SIZE_N, const uint TILE_SIZE_K, const uint ROWS_PER_THREAD>
__global__ void gemm_1D_registertiling_bf16(const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K, float alpha, float beta) {
    // Allocate shared memory
    __shared__ __nv_bfloat16 sharedA[TILE_SIZE_M * TILE_SIZE_N];
    __shared__ __nv_bfloat16 sharedB[TILE_SIZE_N * TILE_SIZE_K];

    // Identify the tile of C this thread block is responsible for 
    const uint block_row = blockIdx.y;
    const uint block_column = blockIdx.x;

    // Calculate position of thread within tile (Remapping from 1-D to 2-D)
    const uint ty = threadIdx.x / TILE_SIZE_K; 
    const uint tx = threadIdx.x % TILE_SIZE_K;

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
    const uint num_tiles = CEIL_DIV(N, TILE_SIZE_N);

    // Initialise results array to match how many elements each thread is handing (This will be stored in register file of each thread)
    float thread_results[ROWS_PER_THREAD] = {0.0f};

    // Iterate over tiles (Phase 1: Loading data)
    for (int t = 0; t < num_tiles; t++) {
        sharedA[smem_ty_A * TILE_SIZE_N + smem_tx_A] = A[smem_ty_A * N + smem_tx_A];
        sharedB[smem_ty_B * TILE_SIZE_K + smem_tx_B] = B[smem_ty_B * K + smem_tx_B];

        // Barrier synchronisation until all threads load smem tiles
        __syncthreads();

        // Loop over the shared dimension between the smem tiles, which is TILE_SIZE_N
        for (int i = 0; i < TILE_SIZE_N; i++) {
            float fixed_B = __bfloat162float(sharedB[i * TILE_SIZE_K + tx]); // Fixate one value from sharedB every iteration
            for (int row = 0; row < ROWS_PER_THREAD; row++) {
                uint global_smem_row_idx = ty * ROWS_PER_THREAD + row;
                thread_results[row] += __bfloat162float(sharedA[global_smem_row_idx * TILE_SIZE_N + i]) * fixed_B;
            }
        }
        // Barrier synchronisation until all threads finish computing their results
        __syncthreads();

        // Move all pointers to the starting positions of the next tile
        A += TILE_SIZE_N; // Move right
        B += TILE_SIZE_N * K; // Move down
    }
    // Write results back to C
    for (int row = 0; row < ROWS_PER_THREAD; row++) {
        uint global_row_idx = ty * ROWS_PER_THREAD + row;
        C[global_row_idx * K + tx] = __float2bfloat16_rn(alpha * thread_results[row] + beta * __bfloat162float(C[global_row_idx * K + tx]));
    }
}


