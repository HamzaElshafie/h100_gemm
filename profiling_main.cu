#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cmath>

#include "utils.h"
#include "kernels/hopper/hopper_tma_utils.h"
#include "kernels/hopper/gemm_bf16_pc_pipeline.cuh"

// Alias for simplicity
using bf16 = __nv_bfloat16;

void launch_gemm_bf16_pc_pipeline(const bf16* A, const bf16* B, bf16* C, int M, int N, int K, float alpha, float beta) {
    constexpr int TILE_SIZE_M = 128;
    constexpr int TILE_SIZE_N = 128;
    constexpr int TILE_SIZE_K = 64;
    constexpr int NUM_THREADS = 128 * 2;
    constexpr int NUM_STAGES = 5;
    constexpr int WGMMA_M = 64;
    constexpr int WGMMA_K = 16;
    constexpr int WGMMA_N = 128;

    CUtensorMap* d_tma_map_A = create_and_allocate_tensor_map<TILE_SIZE_M, TILE_SIZE_K>(
        const_cast<bf16*>(A), CEIL_DIV(M, TILE_SIZE_M), CEIL_DIV(K, TILE_SIZE_K));
    CUtensorMap* d_tma_map_B = create_and_allocate_tensor_map<TILE_SIZE_N, TILE_SIZE_K>(
        const_cast<bf16*>(B), CEIL_DIV(N, TILE_SIZE_N), CEIL_DIV(K, TILE_SIZE_K));

    auto* kernel = gemm_bf16_pc_pipeline<
        TILE_SIZE_M, TILE_SIZE_K, TILE_SIZE_N,
        WGMMA_M, WGMMA_N, WGMMA_K, NUM_THREADS, NUM_STAGES>;
    size_t sMemSize = sizeof(Smem<TILE_SIZE_M, TILE_SIZE_K, TILE_SIZE_N, NUM_STAGES>);
    CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    kernel<<<(M / TILE_SIZE_M) * (N / TILE_SIZE_N), NUM_THREADS, sMemSize>>>(
        d_tma_map_A, d_tma_map_B, C, M, K, N, alpha, beta);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_tma_map_A));
    CUDA_CHECK(cudaFree(d_tma_map_B));
}

/**
 * @brief Main entry point for the profiling program (gemm_bf16_pc_pipeline only).
 */
int main(int argc, char** argv) {
    ResourceManager<bf16> resources;

    // Fixed size for profiling
    const int size = 8192;
    const float alpha = 0.5f;
    const float beta = 3.0f;
    
    const size_t mem_size = size * size * sizeof(bf16);

    std::cout << "Profiling gemm_bf16_pc_pipeline kernel with matrix size " << size << "x" << size << std::endl;

    // Allocate host memory
    bf16* A_host = (bf16*)malloc(mem_size);
    bf16* B_host = (bf16*)malloc(mem_size);
    bf16* C_host = (bf16*)malloc(mem_size);

    // Register host memory with resource manager
    resources.add_host_ptr(A_host);
    resources.add_host_ptr(B_host);
    resources.add_host_ptr(C_host);

    if (!A_host || !B_host || !C_host) {
        std::cerr << "Host memory allocation failed" << std::endl;
        return -1;
    }

    // Initialise matrices
    bf16* matrices[] = {A_host, B_host, C_host};
    initialiseArrays<bf16>(matrices, 3, size * size, -100.0f, 100.0f, 0);

    // Allocate device memory
    bf16* A_device;
    bf16* B_device;
    bf16* C_device;

    CUDA_CHECK(cudaMalloc((void**)&A_device, mem_size));
    CUDA_CHECK(cudaMalloc((void**)&B_device, mem_size));
    CUDA_CHECK(cudaMalloc((void**)&C_device, mem_size));

    resources.add_device_ptr(A_device);
    resources.add_device_ptr(B_device);
    resources.add_device_ptr(C_device);

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(A_device, A_host, mem_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_device, B_host, mem_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C_device, C_host, mem_size, cudaMemcpyHostToDevice));

    std::cout << "Memory allocation and data transfer completed" << std::endl;

    // Warm-up launches
    std::cout << "Running warm-up launches..." << std::endl;
    for (int i = 0; i < 2; ++i) {
        launch_gemm_bf16_pc_pipeline(A_device, B_device, C_device, size, size, size, alpha, beta);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Main kernel launch for profiling
    std::cout << "Running main kernel for profiling..." << std::endl;
    launch_gemm_bf16_pc_pipeline(A_device, B_device, C_device, size, size, size, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Kernel launch completed successfully" << std::endl;

    return 0;  // ResourceManager destructor will handle cleanups
}
