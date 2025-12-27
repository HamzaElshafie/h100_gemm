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
#include "hopper_tma_utils.h"
#include "hopper_wgmma_utils.cuh"
#include "gemm_bf16_wgmma_tma.cuh"

// Alias for simplicity
using bf16 = __nv_bfloat16;

// Global tensor map cache
static CUtensorMap *d_tma_map_A = nullptr;
static CUtensorMap *d_tma_map_B = nullptr;
static int _prev_m = 0, _prev_n = 0, _prev_k = 0;

namespace hopper {
    /**
     * @brief Launches the WGMMA + TMA GEMM kernel with tensor map caching.
     *
     * Manages allocation and caching of tensor map descriptors for efficient repeated
     * kernel launches with the same matrix dimensions.
     */
    void run_gemm_bf16_wgmma_tma(const bf16* __restrict__ A, const bf16* __restrict__ B, bf16* __restrict__ C,
    int M, int N, int K, float alpha, float beta) {
        constexpr int TILE_SIZE_M = 64;
        constexpr int TILE_SIZE_K = 64;
        constexpr int TILE_SIZE_N = 64;
        constexpr int WGMMA_M = 64;
        constexpr int WGMMA_K = 16;
        constexpr int WGMMA_N = 64;
        constexpr int NUM_THREADS = 128;

        // Allocate and create tensor maps on first call or if dimensions change
        if (!d_tma_map_A) {
            d_tma_map_A = create_and_allocate_tensor_map<TILE_SIZE_M, TILE_SIZE_K>(
                const_cast<bf16*>(A), M / TILE_SIZE_M, K / TILE_SIZE_K);
            d_tma_map_B = create_and_allocate_tensor_map<TILE_SIZE_N, TILE_SIZE_K>(
                const_cast<bf16*>(B), N / TILE_SIZE_N, K / TILE_SIZE_K);
            _prev_m = M;
            _prev_n = N;
            _prev_k = K;
        }

        // Assert cached values match current dimensions
        assert(M == _prev_m && N == _prev_n && K == _prev_k && 
               "Matrix dimensions changed; tensor maps are invalid");

        // Calculate grid dimensions
        int num_blocks_m = M / TILE_SIZE_M;
        int num_blocks_n = N / TILE_SIZE_N;
        int grid_size = num_blocks_m * num_blocks_n;

        // Launch kernel with tensor maps
        gemm_bf16_wgmma_tma<TILE_SIZE_M, TILE_SIZE_K, TILE_SIZE_N, WGMMA_M, WGMMA_K, WGMMA_N, NUM_THREADS>
            <<<grid_size, NUM_THREADS>>>(d_tma_map_A, d_tma_map_B, C, M, N, K, alpha, beta);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}