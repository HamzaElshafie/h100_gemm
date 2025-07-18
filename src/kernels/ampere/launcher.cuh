#pragma once

#include <iostream>
#include <cuda_runtime.h>
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
}

namespace cublas {
    void run_sgemm_cublas(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                            int M, int N, int K, float alpha, float beta, cublasHandle_t handle);
}