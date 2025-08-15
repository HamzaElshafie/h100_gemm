#pragma once

#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

/**
 * @brief Naive GEMM for BF16 inputs (accumulates in FP32).
 *
 * Computes the matrix multiplication C = alpha * (A @ B) + beta * C,
 * where A is of size (MxN), B is of size (NxK), and C is of size (MxK).
 * This implementation assigns one thread per element in the output matrix C.
 *
 * @param A       Pointer to input matrix A, stored in row-major order
 * @param B       Pointer to input matrix B
 * @param C       Pointer to output matrix C
 * @param M       Number of rows in matrix A and C
 * @param N       Number of columns in A and rows in B (shared dimension)
 * @param K       Number of columns in matrices B and C
 * @param alpha   Scalar multiplier for the matrix product (A @ B)
 * @param beta    Scalar multiplier for the existing values in matrix C
 */
__global__ void gemm_naive_bf16(const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K, float alpha, float beta) {
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint column = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && column < K) {
        float cumulative_sum = 0.0f;
        for (int n = 0; n < N; n++) { // Loop over shared dimension
            cumulative_sum += __bfloat162float(A[row * N + n]) * __bfloat162float(B[n * K + column]);
        }
        float c_old = __bfloat162float(C[row * K + column]);
        C[row * K + column] = __float2bfloat16_rn(alpha * cumulative_sum + beta * c_old);
    }
}


