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
#include "kernel01_naive.cuh"

namespace simon
{
    /**
     * @brief Launches a naive sgemm kernel
     * 
     * @param[in] A Pointer to the input matrix A of dimensions M x N
     * @param[in] B Pointer to the input matrix B of dimensions N x K
     * @param[out] C Pointer to the output matrix C of dimensions M x K
     * @param[in] M Number of rows in matrix A and matrix C
     * @param[in] N Number of columns in matrix A and rows in matrix B (shared dim)
     * @param[in] K Number of columns in matrix B and matrix C
     * @param[in] alpha Scalar multiplier for the product of matrices A and B
     * @param[in] beta Scalar multiplier for the existing values in matrix C
     */
    void run_sgemm_naive(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, 
        int M, int N, int K, float alpha, float beta)
        {
            // Grid configs
            dim3 gridDim(CEIL_DIV(K, 32), CEIL_DIV(M, 32));
            dim3 blockDim(32, 32);
            sgemm_naive<<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
}