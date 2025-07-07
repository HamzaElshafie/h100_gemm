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
void run_sgemm_cublasLt(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, float alpha, float beta, cublasHandle_t /*unused*/) {

    // Create (once) cuBLASLt handle & workspace
    static cublasLtHandle_t ltHandle = nullptr;
    static void* workspace         = nullptr;
    static const size_t workspace_size = 32 * 1024 * 1024; // 32 MB

    if (ltHandle == nullptr) {
        CUBLAS_CHECK(cublasLtCreate(&ltHandle));
        CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    }

    // --- Create operation descriptor (compute in FP32) ---
    cublasLtMatmulDesc_t operationDesc;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasOperation_t opN = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    // Scaling type (alpha / beta)
    cublasDataType_t scale_type = CUDA_R_32F;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc,
        CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));

    // --- Create matrix layouts (row-major order) ---
    cublasLtMatrixLayout_t ALayout, BLayout, CLayout;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&ALayout, CUDA_R_32F, M, N, N));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&BLayout, CUDA_R_32F, N, K, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&CLayout, CUDA_R_32F, M, K, K));

    int32_t order = CUBLASLT_ORDER_ROW;
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(ALayout,
        CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(BLayout,
        CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(CLayout,
        CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    // --- Set preference & select heuristic algorithm ---
    cublasLtMatmulPreference_t preference;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTE, 
        &workspace_size, sizeof(workspace_size)));

    cublasLtMatmulHeuristicResult_t heuristicResult;
    int returnedResults = 0;
    cublasLtMatmulAlgoGetHeuristic(
        ltHandle,
        operationDesc,
        ALayout, BLayout, CLayout, CLayout,
        preference,
        1,
        &heuristicResult,
        &returnedResults);

    if (returnedResults == 0) {
        std::cerr << "cuBLASLt: No suitable algorithm found for (M,N,K) = ("
                  << M << ", " << N << ", " << K << ")" << std::endl;
        exit(EXIT_FAILURE);
    }

    // --- Perform GEMM ---
    CUBLAS_CHECK(cublasLtMatmul(ltHandle,
                                operationDesc,
                                &alpha,
                                A, ALayout,
                                B, BLayout,
                                &beta,
                                C, CLayout,
                                C, CLayout,
                                &heuristicResult.algo,
                                workspace,
                                workspace_size,
                                /* stream */ 0));

    // Cleanup descriptors (preference/layout/operation)
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(operationDesc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(ALayout));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(BLayout));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(CLayout));

    CUDA_CHECK(cudaDeviceSynchronize());
}
}