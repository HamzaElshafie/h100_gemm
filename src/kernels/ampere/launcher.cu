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
}

namespace cublas {
/**
 * @brief Launches the cuBLAS SGEMM reference kernel using cublasGemmEx.
 *
 * This function uses cuBLAS to perform C = alpha * A * B + beta * C.
 * All matrices are assumed to be in row-major order.
 * cuBLAS expects column-major, so we swap A/B and M/N.
 *
 * @param A       Pointer to input matrix A (device, row-major)
 * @param B       Pointer to input matrix B (device, row-major)
 * @param C       Pointer to output matrix C (device, row-major)
 * @param M       Number of rows of matrix A and C
 * @param N       Number of columns of matrix B and C
 * @param K       Number of columns of matrix A and rows of matrix B
 * @param alpha   Scalar multiplier for the matrix product
 * @param beta    Scalar multiplier for the existing values in matrix C
 * @param handle  cuBLAS handle
 */
void run_sgemm_cublas(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, float alpha, float beta, cublasHandle_t handle) {

    // Use cuBLAS GEMM with Tensor Cores via FP16 inputs and FP32 accumulate
    // Note: Data must be convertible to FP16

    // Convert alpha and beta to void* (they are in FP32)
    const void* alpha_ptr = static_cast<const void*>(&alpha);
    const void* beta_ptr  = static_cast<const void*>(&beta);

    // Launch GEMM: C = alpha * A x B + beta * C
    // All matrices are column-major by default
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
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    CUDA_CHECK(cudaDeviceSynchronize());
}
}