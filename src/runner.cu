/**
 * @file runner.cu
 * @brief Kernel launcher implementation.
 *
 * Dispatches the requested kernel to the launchers.
 */
#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>

#include "runner.cuh"
#include "kernels/simon/launcher.cuh"

/**
 * @brief Launches the selected kernel based on the provided configuration.
 *
 * This function dispatches the call to the appropriate kernel variant based on the configuration.
 *
 * @param config   Kernel configuration specifying type and id
 * @param A        Pointer to input matrix A (row-major)
 * @param B        Pointer to input matrix B
 * @param C        Pointer to output matrix C
 * @param M        Number of rows in matrix A and C
 * @param N        Number of columns in A and rows in B (Shared dimension)
 * @param K        Number of columns in matrices B and C
 * @param alpha    Scalar multiplier for the matrix product (A @ B)
 * @param beta     Scalar multiplier for the existing values in matrix C
 * @param handle   cuBLAS handle for GPU operations
 * @throws std::invalid_argument if the kernel type or ID is unknown
 */
void launchKernel(const KernelConfig& config, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, 
    int M, int N, int K, float alpha, float beta, cublasHandle_t handle) {
        switch(config.type) {
            case KernelType::SIMON:
                switch (static_cast<SimonKernelVariant>(config.kernel_id)) {
                    case SimonKernelVariant::naive_sgemm:
                        simon::run_sgemm_naive(A, B, C, M, N, K, alpha, beta);
                        break;
                    case SimonKernelVariant::coalesced_sgemm:
                        simon::run_sgemm_coalesced(A, B, C, M, N, K, alpha, beta);
                        break;
                    default:
                        throw std::invalid_argument("Unknown Simon kernel ID");
                }
            break;
            case KernelType::CUBLAS:
                // Only one reference so no need for switch here
                cublas::run_sgemm_cublas(A, B, C, M, N, K, alpha, beta, handle);
                break;
            // Hopper TODO
            default:
                throw std::invalid_argument("Unknown kernel type");
        }
    }


