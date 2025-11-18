/**
 * @file launcher.cu
 * @brief Entry point for launching kernels
 * 
 */

#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cmath>

#include "utils.h"
#include "sgemm_naive.cuh"
#include "sgemm_coalesced.cuh"
#include "sgemm_tiled_shared.cuh"
#include "sgemm_1D_registertiling.cuh"
#include "sgemm_2D_registertiling.cuh"
#include "sgemm_vectorised.cuh"
#include "gemm_naive_bf16.cuh"
#include "gemm_coalesced_bf16.cuh"
#include "gemm_tiled_shared_bf16.cuh"
#include "gemm_1D_registertiling_bf16.cuh"
#include "gemm_2D_registertiling_bf16.cuh"
#include "gemm_vectorised_bf16.cuh"
#include "sgemm_warptiling.cuh"
#include "gemm_warptiling_bf16.cuh"

namespace general {
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
    const uint TILE_SIZE_M = 128;
    const uint TILE_SIZE_K = 128;
    dim3 gridDim(CEIL_DIV(K, TILE_SIZE_K), CEIL_DIV(M, TILE_SIZE_M));
    dim3 blockDim((TILE_SIZE_M * TILE_SIZE_K) / (ROWS_PER_THREAD * COLS_PER_THREAD));
    sgemm_2D_registertiling<TILE_SIZE_M, TILE_SIZE_N, TILE_SIZE_K, ROWS_PER_THREAD, COLS_PER_THREAD>
        <<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    }

    void run_sgemm_vectorised(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                                 int M, int N, int K, float alpha, float beta) {
    const uint TILE_SIZE_N = 8;
    const uint ROWS_PER_THREAD = 8;
    const uint COLS_PER_THREAD = 8;
    const uint TILE_SIZE_M = 128;
    const uint TILE_SIZE_K = 128;
    dim3 gridDim(CEIL_DIV(K, TILE_SIZE_K), CEIL_DIV(M, TILE_SIZE_M));
    dim3 blockDim((TILE_SIZE_M * TILE_SIZE_K) / (ROWS_PER_THREAD * COLS_PER_THREAD));
    sgemm_vectorised<TILE_SIZE_M, TILE_SIZE_N, TILE_SIZE_K, ROWS_PER_THREAD, COLS_PER_THREAD>
        <<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    }

    void run_sgemm_warptiling(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
        int M, int N, int K, float alpha, float beta) {
        constexpr int TILE_SIZE_M = 128;
        constexpr int TILE_SIZE_N = 16;
        constexpr int TILE_SIZE_K = 128;
        constexpr int WARP_TILE_M = 64;
        constexpr int WARP_TILE_K = 64;
        constexpr int ROWS_PER_THREAD = 8;
        constexpr int COLS_PER_THREAD = 4;
        constexpr int WARP_STEPS_K = 4;
        constexpr int NUM_THREADS = 128;

        dim3 blockDim(NUM_THREADS);
        dim3 gridDim(CEIL_DIV(K, TILE_SIZE_K), CEIL_DIV(M, TILE_SIZE_M));

        sgemm_warptiling<
            TILE_SIZE_M, TILE_SIZE_N, TILE_SIZE_K,
            WARP_TILE_M, WARP_TILE_K, WARP_STEPS_K,
            ROWS_PER_THREAD, COLS_PER_THREAD, NUM_THREADS>
            <<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void run_gemm_naive_bf16(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C, 
        int M, int N, int K, float alpha, float beta) {
        dim3 gridDim(CEIL_DIV(K, 32), CEIL_DIV(M, 32));
        dim3 blockDim(32, 32);
        gemm_naive_bf16<<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void run_gemm_coalesced_bf16(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C,
        int M, int N, int K, float alpha, float beta) {
        dim3 gridDim(CEIL_DIV(K, 32), CEIL_DIV(M, 32));
        dim3 blockDim(32*32);
        gemm_coalesced_bf16<32><<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void run_gemm_tiled_shared_bf16(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C,
        int M, int N, int K, float alpha, float beta) {
        dim3 gridDim(CEIL_DIV(K, 32), CEIL_DIV(M, 32));
        dim3 blockDim(32*32);
        gemm_tiled_shared_bf16<32><<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void run_gemm_1D_registertiling_bf16(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C,
        int M, int N, int K, float alpha, float beta) {
        const uint TILE_SIZE_M = 64;
        const uint TILE_SIZE_K = 64;
        const uint TILE_SIZE_N = 8;
        const uint ROWS_PER_THREAD = 8;
        dim3 gridDim(CEIL_DIV(K, TILE_SIZE_K), CEIL_DIV(M, TILE_SIZE_M));
        dim3 blockDim((TILE_SIZE_M * TILE_SIZE_K) / ROWS_PER_THREAD);
        gemm_1D_registertiling_bf16<TILE_SIZE_M, TILE_SIZE_N, TILE_SIZE_K, ROWS_PER_THREAD>
            <<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void run_gemm_2D_registertiling_bf16(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C,
        int M, int N, int K, float alpha, float beta) {
        const uint TILE_SIZE_N = 8;
        const uint ROWS_PER_THREAD = 8;
        const uint COLS_PER_THREAD = 8;
        const uint TILE_SIZE_M = 128;
        const uint TILE_SIZE_K = 128;
        dim3 gridDim(CEIL_DIV(K, TILE_SIZE_K), CEIL_DIV(M, TILE_SIZE_M));
        dim3 blockDim((TILE_SIZE_M * TILE_SIZE_K) / (ROWS_PER_THREAD * COLS_PER_THREAD));
        gemm_2D_registertiling_bf16<TILE_SIZE_M, TILE_SIZE_N, TILE_SIZE_K, ROWS_PER_THREAD, COLS_PER_THREAD>
            <<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void run_gemm_vectorised_bf16(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C,
        int M, int N, int K, float alpha, float beta) {
        const uint TILE_SIZE_N = 8;
        const uint ROWS_PER_THREAD = 8;
        const uint COLS_PER_THREAD = 8;
        const uint TILE_SIZE_M = 128;
        const uint TILE_SIZE_K = 128;
        dim3 gridDim(CEIL_DIV(K, TILE_SIZE_K), CEIL_DIV(M, TILE_SIZE_M));
        dim3 blockDim((TILE_SIZE_M * TILE_SIZE_K) / (ROWS_PER_THREAD * COLS_PER_THREAD));
        gemm_vectorised_bf16<TILE_SIZE_M, TILE_SIZE_N, TILE_SIZE_K, ROWS_PER_THREAD, COLS_PER_THREAD>
            <<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void run_gemm_warptiling_bf16(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C,
        int M, int N, int K, float alpha, float beta) {
        constexpr int TILE_SIZE_M = 128;
        constexpr int TILE_SIZE_N = 16;
        constexpr int TILE_SIZE_K = 128;
        constexpr int WARP_TILE_M = 64;
        constexpr int WARP_TILE_K = 64;
        constexpr int ROWS_PER_THREAD = 8;
        constexpr int COLS_PER_THREAD = 4;
        constexpr int WARP_STEPS_K = 4;
        constexpr int NUM_THREADS = 128;

        dim3 blockDim(NUM_THREADS);
        dim3 gridDim(CEIL_DIV(K, TILE_SIZE_K), CEIL_DIV(M, TILE_SIZE_M));

        gemm_warptiling_bf16<
            TILE_SIZE_M, TILE_SIZE_N, TILE_SIZE_K,
            WARP_TILE_M, WARP_TILE_K, WARP_STEPS_K,
            ROWS_PER_THREAD, COLS_PER_THREAD, NUM_THREADS>
            <<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

namespace cublas {
    void run_gemm_cublas(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                           int M, int N, int K, float alpha, float beta, cublasHandle_t handle) {
    // Standard cuBLAS GEMM in full FP32 precision (no tensor cores, no mixed precision)
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose
        K, M, N,                   // Note: cuBLAS is column-major: C[K x M] = B[K x N] × A[N x M]
        &alpha,
        B, CUDA_R_32F, K,         // B: (N x K), lda = K
        A, CUDA_R_32F, N,         // A: (M x N), lda = N
        &beta,
        C, CUDA_R_32F, K,         // C: (M x K), ldc = K
        CUBLAS_COMPUTE_32F_PEDANTIC,  // Strict full FP32 compute, disables tensor cores
        CUBLAS_GEMM_DEFAULT        // Standard algo
    ));

    CUDA_CHECK(cudaDeviceSynchronize());
}
    void run_gemm_cublas_bf16(const __nv_bfloat16* __restrict__ A,
                                const __nv_bfloat16* __restrict__ B,
                                __nv_bfloat16* __restrict__ C,
                                int M, int N, int K, float alpha, float beta, cublasHandle_t handle) {
        // Same row-major -> column-major mapping as the FP32 path.
        // Inputs/outputs are BF16; accumulate in FP32 on Tensor Cores.
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            /* m */ K, /* n */ M, /* k */ N,
            &alpha,
            /* B */ B, CUDA_R_16BF, /* ldb */ K,   // B[K x N]
            /* A */ A, CUDA_R_16BF, /* lda */ N,   // A[N x M]
            &beta,
            /* C */ C, CUDA_R_16BF, /* ldc */ K,   // C[K x M] in column-major == C[M x K] row-major
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}