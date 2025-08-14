#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

namespace hopper {
    void run_gemm_warp_tiling_bf16(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C,
        int M, int N, int K, float alpha, float beta);
}