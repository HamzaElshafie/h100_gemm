#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utils.h"
#include "runner.cuh"

/**
 * @brief Parses the kernel configuration from CLI arguments.
 */
KernelConfig parseKernelConfig(const std::string& impl, int kernel_id) {
    if (impl == "ampere") {
        if (kernel_id > 3 || kernel_id < 0) {
            throw std::invalid_argument("Invalid Ampere kernel ID");
        }
        return KernelConfig(KernelType::AMPERE, kernel_id);
    } else if (impl == "hopper") {
        if (kernel_id > 1 || kernel_id < 0) {
            throw std::invalid_argument("Invalid Hopper kernel ID");
        }
        return KernelConfig(KernelType::HOPPER, kernel_id);
    } else if (impl == "cublas") {
        if (kernel_id != 0) {
            throw std::invalid_argument("Invalid cuBLAS kernel ID (only ID=0 supported)");
        }
        return KernelConfig(KernelType::CUBLAS, kernel_id);
    } else {
        throw std::invalid_argument("Invalid implementation name: " + impl);
    }
}

/**
 * @brief Main entry point for the profiling program.
 *
 * @param argc Number of CLI arguments.
 * @param argv Array of CLI strings.
 * @return int Exit status code.
 */
int main(int argc, char** argv) {
    ResourceManager resources;

    if (argc != 3) {
        std::cout << "Usage: ./profiling_main <implementation> <kernel_ID_number>\n";
        return -1;
    }
    
    std::string impl = argv[1];
    int kernel_id = std::stoi(argv[2]);
    KernelConfig config = parseKernelConfig(impl, kernel_id);
    
    // Fixed size for profiling
    const int size = 8192;
    const float alpha = 0.5f;
    const float beta = 3.0f;
    
    const size_t mem_size = size * size * sizeof(float);

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

    // Create cuBLAS handle if needed
    cublasHandle_t handle;
    if (cublasCreate(&handle)) {
        std::cerr << "Create cublas handle error." << std::endl;
        return -1;
    }
    resources.set_cublas_handle(&handle);

    std::cout << "Running kernel for profiling: " << impl << " ID=" << kernel_id 
              << " Size=" << size << "x" << size << std::endl;

    // Warm-up launches
    for (int i = 0; i < 2; ++i) {
        launchKernel(config, A_device, B_device, C_device, size, size, size, alpha, beta, handle);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Main kernel launch for profiling
    launchKernel(config, A_device, B_device, C_device, size, size, size, alpha, beta, handle);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Kernel launch completed successfully" << std::endl;

    return 0;  // ResourceManager destructor will all handle cleanups
}
