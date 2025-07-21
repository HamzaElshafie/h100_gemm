#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

#include "utils.h"


template <const uint TILE_SIZE>
__global__ void sgemm_tiled_shared(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, float alpha, float beta) {
        // Allocate shared memory
        __shared__ float sharedA[TILE_SIZE * TILE_SIZE];
        __shared__ float sharedB[TILE_SIZE * TILE_SIZE];

        // Identify the tile of C this thread block is responsible for (We assume tiles are same size as block)
        const uint block_row = blockIdx.y;
        const uint block_column = blockIdx.x;

        // Calculate position of thread within tile (Remapping from 1-D to 2-D)
        const uint ty = threadIdx.x / TILE_SIZE; // (0, TILE_SIZE-1)
        const uint tx = threadIdx.x % TILE_SIZE; // (0, TILE_SIZE-1)

        // Move pointers from A[0], B[0] and C[0] to the starting positions of the tile
        A += block_row * TILE_SIZE * N; // Move pointer (block_row * TILE_SIZE) rows down
        B += block_column * TILE_SIZE; // Move pointer (block_column * TILE_SIZE) columns to the right 
        C += (block_row * TILE_SIZE * K) + (block_column * TILE_SIZE); // Move pointer (block_row * TILE_SIZE * K) rows down then (block_column * TILE_SIZE) columns to the right

        // Calculate how many tiles we have
        const uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
        float cumulative_sum = 0.0f;

        // Iterate over tiles (Phase 1: Loading data)
        for (int t = 0; t < num_tiles; t++) {
            sharedA[ty * TILE_SIZE + tx] = A[ty * N + tx];
            sharedB[ty * TILE_SIZE + tx] = B[ty * K + tx];

            // Barrier synchronisation until all threads load smem tiles
            __syncthreads();

            // Phase 2: Compute partial results iteratively
            for (int i = 0; i < TILE_SIZE; i++) {
                cumulative_sum += sharedA[ty * TILE_SIZE + i] * sharedB[i * TILE_SIZE + tx];
            }

            // Barrier synchronisation until all threads finish writing to smem
            __syncthreads();

            // Move all pointers to the starting positions of the next tile
            A += TILE_SIZE; // Move right
            B += TILE_SIZE * K; // Move down
        }
        // Write results back to C
        C[ty * K + tx] = (alpha * cumulative_sum) + (beta * C[ty * K + tx]);
    }

void launch_sgemm_tiled_shared(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta) {
    const int TILE_SIZE = 32;
    dim3 gridDim(CEIL_DIV(K, TILE_SIZE), CEIL_DIV(M, TILE_SIZE));
    dim3 blockDim(TILE_SIZE * TILE_SIZE); // 1024 threads per block
    
    sgemm_tiled_shared<TILE_SIZE><<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * @brief Main entry point for the profiling program.
 */
int main(int argc, char** argv) {
    ResourceManager resources;

    // Fixed size for profiling
    const int size = 8192;
    const float alpha = 0.5f;
    const float beta = 3.0f;
    
    const size_t mem_size = size * size * sizeof(float);

    std::cout << "Profiling sgemm_tiled_shared kernel with matrix size " << size << "x" << size << std::endl;

    // Allocate host memory
    float* A_host = (float*)malloc(mem_size);
    float* B_host = (float*)malloc(mem_size);
    float* C_host = (float*)malloc(mem_size);

    // Register host memory with resource manager
    resources.add_host_ptr(A_host);
    resources.add_host_ptr(B_host);
    resources.add_host_ptr(C_host);

    if (!A_host || !B_host || !C_host) {
        std::cerr << "Host memory allocation failed" << std::endl;
        return -1;
    }

    // Initialise matrices
    float* matrices[] = {A_host, B_host, C_host};
    initialiseArrays(matrices, 3, size * size, -100.0f, 100.0f, 0);

    // Allocate device memory
    float* A_device;
    float* B_device;
    float* C_device;

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
        launch_sgemm_tiled_shared(A_device, B_device, C_device, size, size, size, alpha, beta);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Main kernel launch for profiling
    std::cout << "Running main kernel for profiling..." << std::endl;
    launch_sgemm_tiled_shared(A_device, B_device, C_device, size, size, size, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Kernel launch completed successfully" << std::endl;

    return 0;  // ResourceManager destructor will all handle cleanups
}
