#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <cublas_v2.h>
#include <cuda_bf16.h>

#include "utils.h"
#include "runner.cuh"

/**
 * @brief Prints the usage instructions for the program.
 *
 * This function outputs the correct CLI usage for the executable,
 * including the required arguments and an example.
 */
void printUsage() {
    std::cout << "Usage: ./gemm <implementation> <kernel_ID_number> <dtype>\n"
              << "  Implementation: ampere | hopper | cublas\n"
              << "  ID:       0, 1, 2, ...\n" // TODO: Print last kernel number for each implementation
              << "  dtype:    fp32 | bf16\n"
              << "Example: ./gemm ampere 0 fp32\n"
              << "(Note): For cuBLAS, use ID=0. Example: ./gemm cublas 0 fp32\n";
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
    if (impl == "ampere") { // Check kernel validity
        if (kernel_id > 6 || kernel_id < 0) { // (TODO: Update later)
            throw std::invalid_argument("Invalid Ampere kernel ID");
        }
        return KernelConfig(KernelType::AMPERE, kernel_id);
    } else if (impl == "hopper") {
        if (kernel_id > 1 || kernel_id < 0) { // (TODO: Update later)
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
 * @brief Runs the full logic.
 */
template <typename T>
int run_path(const KernelConfig& config,
             const std::vector<int>& sizes,
             float alpha, float beta) {
    ResourceManager<T> resources; // RAII for pointers, events, and cuBLAS handle

    // Calculate memory size required (Allocate for largest size and reuse for smaller matrices)
    int max_size = sizes.back();
    std::cout << "Max size: " << max_size << std::endl;
    size_t mem_size = static_cast<size_t>(max_size) * static_cast<size_t>(max_size) * sizeof(T);

    cublasHandle_t handle;
    if (cublasCreate(&handle)) {
        std::cerr << "Create cublas handle error." << std::endl;
        return -1;
    }
    resources.set_cublas_handle(&handle);

    // Create events to time trials
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    resources.add_event(start);
    resources.add_event(stop);

    float elapsed_time;

    // Allocate host memory
    T* A_host = (T*)malloc(mem_size);
    T* B_host = (T*)malloc(mem_size);
    T* C_host = (T*)malloc(mem_size);
    T* C_host_ref = (T*)malloc(mem_size);

    // Register host memory with resource manager
    resources.add_host_ptr(A_host);
    resources.add_host_ptr(B_host);
    resources.add_host_ptr(C_host);
    resources.add_host_ptr(C_host_ref);

    if (!A_host || !B_host || !C_host || !C_host_ref) {
        std::cerr << "Host memory allocation failed" << std::endl;
        return -1;  // ResourceManager will clean up automatically
    }

    // Initialise matrices
    T* matrices[] = {A_host, B_host, C_host};
    initialiseArrays<T>(matrices, 3, static_cast<size_t>(max_size) * static_cast<size_t>(max_size), -1.0f, 1.0f, 0);

    // Allocate device memory
    T* A_device;
    T* B_device;
    T* C_device;
    T* C_device_ref;

    CUDA_CHECK(cudaMalloc((void**)&A_device, mem_size));
    CUDA_CHECK(cudaMalloc((void**)&B_device, mem_size));
    CUDA_CHECK(cudaMalloc((void**)&C_device, mem_size));
    CUDA_CHECK(cudaMalloc((void**)&C_device_ref, mem_size));

    // Register device memory
    resources.add_device_ptr(A_device);
    resources.add_device_ptr(B_device);
    resources.add_device_ptr(C_device);
    resources.add_device_ptr(C_device_ref);

    int repeat = 50;
    for (int size : sizes) {
        int M = size;
        int N = size;
        int K = size;

        // Calculate current memory size required
        size_t curr_mem_size = static_cast<size_t>(size) * static_cast<size_t>(size) * sizeof(T);

        std::cout << "Dimensions (M = N = K) = " << M << " Alpha: " << alpha << " Beta: " << beta << std::endl;

        // Copy data from host to device
        CUDA_CHECK(cudaMemcpy(A_device, A_host, curr_mem_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(B_device, B_host, curr_mem_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(C_device, C_host, curr_mem_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(C_device_ref, C_host, curr_mem_size, cudaMemcpyHostToDevice));

        // Run cuBLAS and custom kernel to check for correctness and warmup
        if (config.type != KernelType::CUBLAS) {
            // Custom op
            launchKernel<T>(config, A_device, B_device, C_device, M, N, K, alpha, beta, handle);
            CUDA_CHECK(cudaMemcpy(C_host, C_device, curr_mem_size, cudaMemcpyDeviceToHost));

            // cuBLAS op
            KernelConfig cublas_config(KernelType::CUBLAS, 0);
            launchKernel<T>(cublas_config, A_device, B_device, C_device_ref, M, N, K, alpha, beta, handle);
            CUDA_CHECK(cudaMemcpy(C_host_ref, C_device_ref, curr_mem_size, cudaMemcpyDeviceToHost));

            // Verify results
            bool results_match = compareResults<T>(C_host_ref, C_host, static_cast<size_t>(M) * static_cast<size_t>(K), 2e-1f, 2e-1f);
            if (!results_match) {
                std::cout << "Results do not match!" << std::endl;
                return -1;
            } else {
                std::cout << "Results match!" << std::endl;
            }
        }

        // Calculate total FLOPs for GEMM: (2*M*N*K + 3*M*N) for alpha*(AB) + beta*C
        double flops_per_run = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K) +
                               3.0 * static_cast<double>(M) * static_cast<double>(N);

        // Warmup cuBLAS kernel
        KernelConfig cublas_config(KernelType::CUBLAS, 0);
        launchKernel<T>(cublas_config, A_device, B_device, C_device_ref, M, N, K, alpha, beta, handle);

        // Start cuBLAS timing
        CUDA_CHECK(cudaEventRecord(start));
        // Run kernel multiple time to smooth out timing variations
        for (int i = 0; i < repeat; i++) {
            launchKernel<T>(cublas_config, A_device, B_device, C_device, M, N, K, alpha, beta, handle);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

        elapsed_time /= 1000.;
        double cublas_avg_time = elapsed_time / repeat;
        double cublas_tflops = flops_per_run / (cublas_avg_time * 1e12);

        // Start custom kernel timing step
        CUDA_CHECK(cudaEventRecord(start));
        // Run kernel multiple time to smooth out timing variations
        for (int i = 0; i < repeat; i++) {
            launchKernel<T>(config, A_device, B_device, C_device, M, N, K, alpha, beta, handle);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

        elapsed_time /= 1000.; // Convert to seconds
        double average_time = elapsed_time / repeat;
        // Throughput in TFLOPs/s
        double custom_tflops = flops_per_run / (average_time * 1e12);

        // Performance relative to cuBLAS
        double perf_ratio = custom_tflops / cublas_tflops;

        printf("Average elapsed time: %.6f s, TFLOPS: %.1f, cuBLAS TFLOPS: %.1f, Performance relative to cuBLAS: %.1f%%\n",
               average_time, custom_tflops, cublas_tflops, perf_ratio * 100.0);

        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(C_host, C_device, curr_mem_size, cudaMemcpyDeviceToHost));
    }

    return 0;  // ResourceManager destructor will handle cleanups
}

/**
 * @brief Main entry point for the program.
 *
 * @param argc Number of CLI arguments.
 * @param argv Array of CLI strings.
 * @return int Exit status code.
 */
int main(int argc, char** argv) {

    if (argc != 4) {
        printUsage();
        return -1;
    }

    std::string impl = argv[1];
    int kernel_id = std::stoi(argv[2]);
    std::string dtype_str = argv[3];
    KernelConfig config = parseKernelConfig(impl, kernel_id);

    // Define matrices sizes to test
    std::vector<int> sizes = {512, 1024, 2048, 4096, 8192};
    float alpha = 0.5f;
    float beta = 3.0f;

    if (dtype_str == "fp32") {
        return run_path<float>(config, sizes, alpha, beta);
    } else if (dtype_str == "bf16") {
        return run_path<__nv_bfloat16>(config, sizes, alpha, beta);
    } else {
        std::cerr << "Invalid dtype. Use 'fp32' or 'bf16'." << std::endl;
        printUsage();
        return -1;
    }
}