#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>

#include "utils.h"
#include "runner.cuh"

/**
 * @brief Prints the usage instructions for the program.
 *
 * This function outputs the correct CLI usage for the executable,
 * including the required arguments and an example.
 */
void printUsage() {
    std::cout << "Usage: ./sgemm <implementation> <kernel_ID_number>\n"
              << "  implementation: simon | hopper\n"
              << "  ID:       0, 1, 2, ...\n" // TODO: Print last kernel number for each implementation
              << "Example: ./sgemm simon 0\n";
}

/**
 * @brief Parses the kernel configuration from CLI arguments.
 *
 * @param impl      The implementation name (e.g., "simon" or "hopper").
 * @param kernel_id The kernel variant number.
 * @return KernelConfig The parsed kernel configuration.
 * @throws std::invalid_argument if the kernel_id is invalid.
 */
KernelConfig parseKernelConfig(const std::string& impl, int kernel_id) {
    if (impl == "simon") { // Check kernel validity
        if (kernel_id > 1 || kernel_id < 0) { // (TODO: Update later)
            throw std::invalid_argument("Invalid Simon kernel ID");
        }
        return KernelConfig(KernelType::SIMON, kernel_id);
    } else if (impl == "hopper") {
        if (kernel_id > 1 || kernel_id < 0) { // (TODO: Update later)
            throw std::invalid_argument("Invalid Hopper kernel ID");
        }
        return KernelConfig(KernelType::HOPPER, kernel_id);
    } else if (impl == "cublas") {
        if (kernel_id != 0) {
            throw std::invalid_argument("Invalid cuBLAS kernel ID"); // Only 0
        }
        return KernelConfig(KernelType::CUBLAS, kernel_id);
    }
}

/**
 * @brief Main entry point for the program.
 *
 * @param argc Number of CLI arguments.
 * @param argv Array of CLI strings.
 * @return int Exit status code.
 */
int main(int argc, char** argv) {
    if (argc != 3) {
        printUsage();
        return -1;
    }
    try {
        std::string impl = argv[1];
        int kernel_id = std::stoi(argv[2]);
        KernelConfig config = parseKernelConfig(impl, kernel_id);
        
        // Define matrices sizes to test
        std::vector<int> sizes = {128, 256, 512, 1024, 2048};
        float alpha = 1.0f;
        float beta = 1.0f;

        // Calculate memory size required (Allocate for largest size and reuse for smaller matrices)
        int max_size = sizes.back();
        std::cout << "Max size: " << max_size << std::endl;
        size_t size = max_size * max_size * sizeof(float);

        // Allocate host memory
        float* A_host = (float*)malloc(size);
        float* B_host = (float*)malloc(size);
        float* C_host = (float*)malloc(size);
        float* C_host_ref = (float*)malloc(size);

        // Initialise matrices
        float* matrices[] = {A_host, B_host, C_host};
        initialiseArrays(matrices, 3, max_size * max_size, -100.0f, 100.0f, 0);

        // Allocate device memory
        float* A_device;
        float* B_device;
        float* C_device;
        float* C_device_ref;

        CUDA_CHECK(cudaMalloc((void**)&A_device, size));
        CUDA_CHECK(cudaMalloc((void**)&B_device, size));
        CUDA_CHECK(cudaMalloc((void**)&C_device, size));
        CUDA_CHECK(cudaMalloc((void**)&C_device_ref, size));

        // Copy data from host to device
        CUDA_CHECK(cudaMemcpy(A_device, A_host, size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(B_device, B_host, size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(C_device, C_host, size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(C_device_ref, C_host, size, cudaMemcpyHostToDevice));

        cublasHandle_t handle;
        if (cublasCreate(&handle)) {
            std::cerr << "Create cublas handle error." << std::endl;
            return -1;
        };

        // Create events to time trials
        cudaEvent_t start;
        cudaEvent_t end;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int repeat = 50;
        for (int size: size) {
            int M = size;
            int N = size;
            int K = size;

            // TODO
        }

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}

