#pragma once

#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

// Alias for simplicity
using bf16 = __nv_bfloat16;

template <const uint TILE_SIZE_M, const uint TILE_SIZE_K, const uint TILE_SIZE_N,
          const uint WGMMA_M, const uint WGMMA_K, const uint WGMMA_N, const uint NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
gemm_bf16_wgmma_tma(const CUtensorMap* __restrict__ tensorMapA, const CUtensorMap* __restrict__ tensorMapB, bf16* __restrict__ C,
    int M, int N, int K, float alpha, float beta) {
    // TODO
}