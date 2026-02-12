#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_bf16.h>

// Alias for simplicity
using bf16 = __nv_bfloat16;

namespace hopper {

    void run_gemm_bf16_wgmma_tma(const bf16 *__restrict__ A, const bf16 *__restrict__ B, bf16 *__restrict__ C,
                                 int M, int N, int K, float alpha, float beta);
    void run_gemm_bf16_wgmma_tma_shapes(const bf16 *__restrict__ A, const bf16 *__restrict__ B, bf16 *__restrict__ C,
                                        int M, int N, int K, float alpha, float beta);
    void run_gemm_bf16_pc_pipeline(const bf16 *__restrict__ A, const bf16 *__restrict__ B, bf16 *__restrict__ C,
                                  int M, int N, int K, float alpha, float beta);
}
