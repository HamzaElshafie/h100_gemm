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
#include "simon_naive.cuh"
#include "simon_coalesced.cuh"

namespace simon {
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

    void run_sgemm_coalesced(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
        int M, int N, int K, float alpha, float beta) {
            dim3 gridDim(CEIL_DIV(K, 32), CEIL_DIV(M, 32));
            dim3 blockDim(32*32);
            sgemm_coalesced<32><<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
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
    // cuBLAS uses column-major order. So we change the order of our row-major A & B,
    // since (B^T*A^T)^T = (A*B)
    cublasStatus_t stat = cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K, // Note: N and M are swapped for row-major
        &alpha,
        B, CUDA_R_32F, N, // B is on the left in column-major
        A, CUDA_R_32F, K,
        &beta,
        C, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS cublasGemmEx failed!" << std::endl;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}
}