/**
 * @file runner.cuh
 * @brief Kernel configuration types and kernel launcher declarations.
 */

#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "launcher.cuh"

/**
 * @brief Types of supported kernel implementations.
 */
enum class KernelType {
    SIMON,
    HOPPER,
    CUBLAS
};

/**
 * @brief Simon kernel variants.
 */
enum class SimonKernelVariant {
    naive_sgemm = 0
};

/**
 * @brief Hopper kernel variants (to be defined).
 */
enum class HopperKernelVariant {
    // TODO
};

/**
 * @brief Configuration for selecting and launching a specific kernel.
 */
struct KernelConfig {
    KernelType type;   /**< The type of kernel implementation (Simon, Hopper or CUBLAS) */
    int kernel_id;     /**< The kernel variant ID */

    /**
     * @brief Construct a new KernelConfig object.
     * @param t Kernel type (Simon or Hopper)
     * @param id Kernel variant ID
     */
    KernelConfig(KernelType t, int id) : type(t), kernel_id(id) {}
};

void launchKernel(const KernelConfig& config, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, 
    int M, int N, int K, float alpha, float beta, cublasHandle_t handle);

