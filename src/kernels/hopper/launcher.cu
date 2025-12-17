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
#include "gemm_bf16_wgmma_tma.cuh"

namespace hopper {
    /**
     * @brief Launches the tensor cores kernel.
     */
    void run_gemm_bf16_wgmma_tma(const CUtensorMap* __restrict__ tensorMapA, const CUtensorMap* __restrict__ tensorMapB, bf16* __restrict__ C,
    int M, int N, int K, float alpha, float beta) {
        // TODO
    }
}