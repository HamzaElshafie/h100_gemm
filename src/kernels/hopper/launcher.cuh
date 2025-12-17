#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

namespace hopper {
    
    void run_gemm_bf16_wgmma_tma(const CUtensorMap *__restrict__ tensorMapA, const CUtensorMap *__restrict__ tensorMapB, bf16 *__restrict__ C,
                                 int M, int N, int K, float alpha, float beta);
}