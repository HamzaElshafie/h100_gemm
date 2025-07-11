#pragma once

#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>

/**
 * @brief Tiled Shared Memory SGEMM (Single-Precision General Matrix Multiply) kernel.
 *
 * Computes the matrix multiplication C = alpha * (A @ B) + beta * C,
 * where A is of size (MxN), B is of size (NxK), and C is of size (MxK).
 * This implementation uses tiled shared memory for improved performance,
 * where each thread block computes a tile of the output matrix C, allowing
 * for efficient data reuse and reduced global memory accesses.
 *
 * @tparam TILE_SIZE Size of the tiles used in the computation
 * @param A       Pointer to input matrix A, stored in row-major order
 * @param B       Pointer to input matrix B
 * @param C       Pointer to output matrix C
 * @param M       Number of rows in matrix A and C
 * @param N       Number of columns in A and rows in B (shared dimension)
 * @param K       Number of columns in matrices B and C
 * @param alpha   Scalar multiplier for the matrix product (A @ B)
 * @param beta    Scalar multiplier for the existing values in matrix C
 */
template <const uint TILE_SIZE>
__global__ void sgemm_tiled_shared(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, float alpha, float beta) {
        // Allocate shared memory
        __shared__ float sharedA[TILE_SIZE * TILE_SIZE];
        __shared__ float sharedB[TILE_SIZE * TILE_SIZE];

        // Identify the tile of C this thread block is responsible for (We assume tiles are same size as block)
        const uint block_row = blockIdx.y;
        const uint block_column = blockIdx.x;

        // Calculate position of thread within tile (Remapping from 1-D to 2-D)
        const uint ty = threadIdx.x / TILE_SIZE; // (0, TILE_SIZE-1)
        const uint tx = threadIdx.x % TILE_SIZE; // (0, TILE_SIZE-1)

        // Move pointers from A[0], B[0] and C[0] to the starting positions of the tile
        A += block_row * TILE_SIZE * N; // Move pointer (block_row * TILE_SIZE) rows down
        B += block_column * TILE_SIZE; // Move pointer (block_column * TILE_SIZE) columns to the right 
        C += (block_row * TILE_SIZE * K) + (block_column * TILE_SIZE); // Move pointer (block_row * TILE_SIZE * K) rows down then (block_column * TILE_SIZE) columns to the right

        // Calculate how many tiles we have
        const uint num_tiles = CEIL_DIV(K, TILE_SIZE);
        float cumulative_sum = 0.0f;

        // Iterate over tiles (Phase 1: Loading data)
        for (int t = 0; t < num_tiles; t++) {
            sharedA[ty * TILE_SIZE + tx] = A[ty * N + tx];
            sharedB[ty * TILE_SIZE + tx] = B[ty * K + tx];

            // Barrier synchronisation until all threads load smem tiles
            __syncthreads();

            // Phase 2: Compute partial results iteratively
            for (int i = 0; i < TILE_SIZE; i++) {
                cumulative_sum += sharedA[ty * TILE_SIZE + i] * sharedB[i * TILE_SIZE + tx];
            }

            // Barrier synchronisation until all threads finish writing to smem
            __syncthreads();

            // Move all pointers to the starting positions of the next tile
            A += TILE_SIZE; // Move right
            B += TILE_SIZE * K; // Move down
        }
        // Write results back to C
        C[ty * K + tx] = (alpha * cumulative_sum) + (beta * C[ty * K + tx]);
    }