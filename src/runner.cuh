/**
 * @file runner.cuh
 * @brief Kernel configuration types and kernel launcher declarations.
 */

#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "kernels/ampere/launcher.cuh"

/**
 * @brief Types of supported kernel implementations.
 */
enum class KernelType {
    AMPERE,
    HOPPER,
    CUBLAS
};

/**
 * @brief Ampere kernel variants.
 */
enum class AmpereKernelVariant {
    naive_sgemm = 0,
    coalesced_sgemm = 1,
    sgemm_tiled_shared = 2,
    sgemm_1D_blocktiling = 3
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
    KernelType type;   /**< The type of kernel implementation (Ampere, Hopper or CUBLAS) */
    int kernel_id;     /**< The kernel variant ID */

    /**
     * @brief Construct a new KernelConfig object.
     * @param t Kernel type (Ampere or Hopper)
     * @param id Kernel variant ID
     */
    KernelConfig(KernelType t, int id) : type(t), kernel_id(id) {}
};

void launchKernel(const KernelConfig& config, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, 
    int M, int N, int K, float alpha, float beta, cublasHandle_t handle);

