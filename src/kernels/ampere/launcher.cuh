#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

namespace ampere {
    void run_sgemm_naive(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, 
        int M, int N, int K, float alpha, float beta);

    void run_sgemm_coalesced(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
        int M, int N, int K, float alpha, float beta);

    void run_sgemm_tiled_shared(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
        int M, int N, int K, float alpha, float beta);

    void run_sgemm_1D_registertiling(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
        int M, int N, int K, float alpha, float beta);

    void run_sgemm_2D_registertiling(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
        int M, int N, int K, float alpha, float beta);

    void run_sgemm_vectorised(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
        int M, int N, int K, float alpha, float beta);

    void run_sgemm_warptiling(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
        int M, int N, int K, float alpha, float beta);
        
    void run_gemm_warptiling_bf16(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C,
        int M, int N, int K, float alpha, float beta);

    void run_gemm_naive_bf16(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C,
        int M, int N, int K, float alpha, float beta);

    void run_gemm_coalesced_bf16(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C,
        int M, int N, int K, float alpha, float beta);

    void run_gemm_tiled_shared_bf16(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C,
        int M, int N, int K, float alpha, float beta);

    void run_gemm_1D_registertiling_bf16(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C,
        int M, int N, int K, float alpha, float beta);

    void run_gemm_2D_registertiling_bf16(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C,
        int M, int N, int K, float alpha, float beta);

    void run_gemm_vectorised_bf16(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C,
        int M, int N, int K, float alpha, float beta);
}

namespace cublas {
    void run_gemm_cublas(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
        int M, int N, int K, float alpha, float beta, cublasHandle_t handle);

    void run_gemm_cublas_bf16(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C,
        int M, int N, int K, float alpha, float beta, cublasHandle_t handle);
}