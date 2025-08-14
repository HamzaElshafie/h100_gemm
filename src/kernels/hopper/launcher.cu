/**
 * @file launcher.cu
 * @brief Entry point for launching kernels
 * 
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cassert>
#include "utils.h"
#include "gemm_warptiling_bf16.cuh"  // your kernel header

namespace hopper {
void run_gemm_warp_tiling_bf16(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C,
    int M, int N, int K, float alpha, float beta)
{
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