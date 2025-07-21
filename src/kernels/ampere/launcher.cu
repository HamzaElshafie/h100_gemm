/**
 * @file launcher.cu
 * @brief Entry point for launching kernels
 * 
 */

#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

#include "utils.h"
#include "sgemm_naive.cuh"
#include "sgemm_coalesced.cuh"
#include "sgemm_tiled_shared.cuh"
#include "sgemm_1D_registertiling.cuh"
#include "sgemm_2D_registertiling.cuh"

namespace ampere {
    /**
     * @brief Launches a naive sgemm kernel
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
    void run_sgemm_naive(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, 
        int M, int N, int K, float alpha, float beta) {
            // Grid configs
            dim3 gridDim(CEIL_DIV(K, 32), CEIL_DIV(M, 32));
            dim3 blockDim(32, 32);
            sgemm_naive<<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

    /**
     * @brief Launches a coalesced sgemm kernel
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
    void run_sgemm_coalesced(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
        int M, int N, int K, float alpha, float beta) {
            dim3 gridDim(CEIL_DIV(K, 32), CEIL_DIV(M, 32));
            dim3 blockDim(32*32); // 1024 threads per block
            sgemm_coalesced<32><<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

    void run_sgemm_tiled_shared(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, float alpha, float beta) {
        dim3 gridDim(CEIL_DIV(K, 32), CEIL_DIV(M, 32));
        dim3 blockDim(32*32); // 1024 threads per block
        sgemm_tiled_shared<32><<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void run_sgemm_1D_registertiling(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, float alpha, float beta) {
        const uint TILE_SIZE_M = 64;
        const uint TILE_SIZE_K = 64;
        const uint TILE_SIZE_N = 8;
        const uint ROWS_PER_THREAD = 8;
        dim3 gridDim(CEIL_DIV(K, TILE_SIZE_K), CEIL_DIV(M, TILE_SIZE_M));
        dim3 blockDim((TILE_SIZE_M * TILE_SIZE_K) / ROWS_PER_THREAD);
        sgemm_1D_registertiling<TILE_SIZE_M, TILE_SIZE_N, TILE_SIZE_K, ROWS_PER_THREAD>
            <<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void run_sgemm_2D_registertiling(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                                 int M, int N, int K, float alpha, float beta) {
    const uint TILE_SIZE_N = 8;
    const uint ROWS_PER_THREAD = 8;
    const uint COLS_PER_THREAD = 8;

    if (M >= 128 && K >= 128) {
        const uint TILE_SIZE_M = 128;
        const uint TILE_SIZE_K = 128;
        dim3 gridDim(CEIL_DIV(K, TILE_SIZE_K), CEIL_DIV(M, TILE_SIZE_M));
        dim3 blockDim((TILE_SIZE_M * TILE_SIZE_K) / (ROWS_PER_THREAD * COLS_PER_THREAD));
        sgemm_2D_registertiling<TILE_SIZE_M, TILE_SIZE_N, TILE_SIZE_K, ROWS_PER_THREAD, COLS_PER_THREAD>
            <<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
    } else {
        // fallback to smaller tile
        const uint TILE_SIZE_M = 64;
        const uint TILE_SIZE_K = 64;
        dim3 gridDim(CEIL_DIV(K, TILE_SIZE_K), CEIL_DIV(M, TILE_SIZE_M));
        dim3 blockDim((TILE_SIZE_M * TILE_SIZE_K) / (ROWS_PER_THREAD * COLS_PER_THREAD));
        sgemm_2D_registertiling<TILE_SIZE_M, TILE_SIZE_N, TILE_SIZE_K, ROWS_PER_THREAD, COLS_PER_THREAD>
            <<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    }
}

namespace cublas {
//     void run_sgemm_cublas(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
//                            int M, int N, int K, float alpha, float beta, cublasHandle_t handle) {
//     // Standard cuBLAS GEMM in full FP32 precision (no tensor cores, no mixed precision)
//     CUBLAS_CHECK(cublasGemmEx(
//         handle,
//         CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose
//         K, M, N,                   // Note: cuBLAS is column-major: C[K x M] = B[K x N] × A[N x M]
//         &alpha,
//         B, CUDA_R_32F, K,         // B: (N x K), lda = K
//         A, CUDA_R_32F, N,         // A: (M x N), lda = N
//         &beta,
//         C, CUDA_R_32F, K,         // C: (M x K), ldc = K
//         CUBLAS_COMPUTE_32F_PEDANTIC,  // Strict full FP32 compute, disables tensor cores
//         CUBLAS_GEMM_DEFAULT        // Standard algo
//     ));

//     CUDA_CHECK(cudaDeviceSynchronize());
// }

    void run_sgemm_cublas(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, float alpha, float beta, cublasHandle_t handle) {

    const void* alpha_ptr = static_cast<const void*>(&alpha);
    const void* beta_ptr  = static_cast<const void*>(&beta);

    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,  // no transpose
        K, M, N,                   // cuBLAS uses column-major: compute C[K×M] = A[K×N] × B[N×M]
        alpha_ptr,
        B, CUDA_R_16F, K,         // B: (N x K), lda = K
        A, CUDA_R_16F, N,         // A: (M x N), lda = N
        beta_ptr,
        C, CUDA_R_32F, K,         // C: (M x K), ldc = K
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP // Uses tensor cores
    ));
}
}