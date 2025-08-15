/**
 * @file runner.cuh
 * @brief Kernel configuration types and kernel launcher declarations.
 */

#pragma once

#include <type_traits>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <stdexcept> 

#include "kernels/ampere/launcher.cuh"
#include "kernels/hopper/launcher.cuh"

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
    sgemm_1D_registertiling = 3,
    sgemm_2D_registertiling = 4,
    sgemm_vectorised = 5,
    sgemm_warptiling = 6,
    // Aliases so the same ID can be used regardless of dtype choice (fp32 or bf16)
    gemm_naive_bf16 = naive_sgemm,
    gemm_coalesced_bf16 = coalesced_sgemm,
    gemm_tiled_shared_bf16 = sgemm_tiled_shared,
    gemm_1D_registertiling_bf16 = sgemm_1D_registertiling,
    gemm_2D_registertiling_bf16 = sgemm_2D_registertiling,
    gemm_vectorised_bf16 = sgemm_vectorised,
    gemm_warptiling_bf16 = sgemm_warptiling
};

/**
 * @brief Hopper kernel variants (to be defined).
 */
enum class HopperKernelVariant {
    gemm_warptiling_bf16 = 0,
    gemm_warptiling = gemm_warptiling_bf16  // Alias so the same ID can be used regardless of dtype choice (fp32 or bf16)
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

template <typename>
struct always_false : std::false_type {};

/**
 * @brief Launches the selected kernel based on the provided configuration.
 *
 * Template dispatches on element type T (float or __nv_bfloat16) so that the same
 * Hopper variant ID can be used for both fp32 and bf16 runs.
 */
template <typename T>
void launchKernel(const KernelConfig& config,
                  const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C, 
                  int M, int N, int K, float alpha, float beta, cublasHandle_t handle) {

    if constexpr (std::is_same_v<T, float>) {
        switch (config.type) {
            case KernelType::AMPERE:
                switch (static_cast<AmpereKernelVariant>(config.kernel_id)) {
                    case AmpereKernelVariant::naive_sgemm:
                        ampere::run_sgemm_naive(A, B, C, M, N, K, alpha, beta);
                        break;
                    case AmpereKernelVariant::coalesced_sgemm:
                        ampere::run_sgemm_coalesced(A, B, C, M, N, K, alpha, beta);
                        break;
                    case AmpereKernelVariant::sgemm_tiled_shared:
                        ampere::run_sgemm_tiled_shared(A, B, C, M, N, K, alpha, beta);
                        break;
                    case AmpereKernelVariant::sgemm_1D_registertiling:
                        ampere::run_sgemm_1D_registertiling(A, B, C, M, N, K, alpha, beta);
                        break;
                    case AmpereKernelVariant::sgemm_2D_registertiling:
                        ampere::run_sgemm_2D_registertiling(A, B, C, M, N, K, alpha, beta);
                        break;
                    case AmpereKernelVariant::sgemm_vectorised:
                        ampere::run_sgemm_vectorised(A, B, C, M, N, K, alpha, beta);
                        break;
                    case AmpereKernelVariant::sgemm_warptiling:
                        ampere::run_sgemm_warptiling(A, B, C, M, N, K, alpha, beta);
                        break;
                    default:
                        throw std::invalid_argument("Unknown Ampere kernel ID");
                }
                break;

            case KernelType::HOPPER:
                throw std::invalid_argument("No generic Hopper-only kernels here; use CUBLAS or Ampere path for architecture-agnostic kernels");

            case KernelType::CUBLAS:
                cublas::run_gemm_cublas(A, B, C, M, N, K, alpha, beta, handle);
                break;
        }

    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        switch (config.type) {
            case KernelType::HOPPER:
                throw std::invalid_argument("No generic Hopper-only kernels here; use CUBLAS or Ampere path for architecture-agnostic kernels");

            case KernelType::CUBLAS:
                cublas::run_gemm_cublas_bf16(A, B, C, M, N, K, alpha, beta, handle);
                break;

            case KernelType::AMPERE:
                // Same variant ID, bf16 path calls the bf16 kernels
                switch (static_cast<AmpereKernelVariant>(config.kernel_id)) {
                    case AmpereKernelVariant::naive_sgemm:
                        ampere::run_gemm_naive_bf16(A, B, C, M, N, K, alpha, beta);
                        break;
                    case AmpereKernelVariant::coalesced_sgemm:
                        ampere::run_gemm_coalesced_bf16(A, B, C, M, N, K, alpha, beta);
                        break;
                    case AmpereKernelVariant::sgemm_tiled_shared:
                        ampere::run_gemm_tiled_shared_bf16(A, B, C, M, N, K, alpha, beta);
                        break;
                    case AmpereKernelVariant::sgemm_1D_registertiling:
                        ampere::run_gemm_1D_registertiling_bf16(A, B, C, M, N, K, alpha, beta);
                        break;
                    case AmpereKernelVariant::sgemm_2D_registertiling:
                        ampere::run_gemm_2D_registertiling_bf16(A, B, C, M, N, K, alpha, beta);
                        break;
                    case AmpereKernelVariant::sgemm_vectorised:
                        ampere::run_gemm_vectorised_bf16(A, B, C, M, N, K, alpha, beta);
                        break;
                    case AmpereKernelVariant::sgemm_warptiling:
                        ampere::run_gemm_warptiling_bf16(A, B, C, M, N, K, alpha, beta);
                        break;
                    default:
                        throw std::invalid_argument("Unknown Ampere kernel ID");
                }
        }
    } else {
        static_assert(always_false<T>::value, "Unsupported element type for launchKernel");
    }
}