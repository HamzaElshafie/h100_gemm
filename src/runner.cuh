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

#include "kernels/general/launcher.cuh"
#include "kernels/hopper/launcher.cuh"

/**
 * @brief Types of supported kernel implementations.
 */
enum class KernelType {
    GENERAL,
    HOPPER,
    CUBLAS
};

/**
 * @brief general kernel variants.
 */
enum class GeneralKernelVariant {
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
 * @brief Hopper kernel variants.
 */
enum class HopperKernelVariant {
    gemm_bf16_wgmma_tma = 0,
    gemm_bf16_wgmma_tma_shapes = 1,
    gemm_bf16_pc_pipeline = 2
};

/**
 * @brief Configuration for selecting and launching a specific kernel.
 */
struct KernelConfig {
    KernelType type; /**< The type of kernel implementation (general, Hopper or CUBLAS) */
    int kernel_id;   /**< The kernel variant ID */

    /**
     * @brief Construct a new KernelConfig object.
     * @param t Kernel type (general or Hopper)
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
 * general variant ID can be used for both fp32 and bf16 runs.
 * 
 * @param comparison_type When config.type is CUBLAS and T is bf16, this determines
 *                        which cuBLAS variant to use (GENERAL or HOPPER layout).
 *                        Ignored for other cases.
 */
template <typename T>
void launchKernel(const KernelConfig &config,
                  const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C,
                  int M, int N, int K, float alpha, float beta, cublasHandle_t handle,
                  KernelType comparison_type = KernelType::GENERAL){

    if constexpr (std::is_same_v<T, float>){
        switch (config.type) {
        case KernelType::GENERAL:
            switch (static_cast<GeneralKernelVariant>(config.kernel_id)){
            case GeneralKernelVariant::naive_sgemm:
                general::run_sgemm_naive(A, B, C, M, N, K, alpha, beta);
                break;
            case GeneralKernelVariant::coalesced_sgemm:
                general::run_sgemm_coalesced(A, B, C, M, N, K, alpha, beta);
                break;
            case GeneralKernelVariant::sgemm_tiled_shared:
                general::run_sgemm_tiled_shared(A, B, C, M, N, K, alpha, beta);
                break;
            case GeneralKernelVariant::sgemm_1D_registertiling:
                general::run_sgemm_1D_registertiling(A, B, C, M, N, K, alpha, beta);
                break;
            case GeneralKernelVariant::sgemm_2D_registertiling:
                general::run_sgemm_2D_registertiling(A, B, C, M, N, K, alpha, beta);
                break;
            case GeneralKernelVariant::sgemm_vectorised:
                general::run_sgemm_vectorised(A, B, C, M, N, K, alpha, beta);
                break;
            case GeneralKernelVariant::sgemm_warptiling:
                general::run_sgemm_warptiling(A, B, C, M, N, K, alpha, beta);
                break;
            default:
                throw std::invalid_argument("Unknown general kernel ID");
            }
            break;

        case KernelType::HOPPER:
            throw std::invalid_argument("No Hopper-only kernels here (No support for FP32 dtype); use general path for architecture-agnostic kernels");

        case KernelType::CUBLAS:
            cublas::run_gemm_cublas(A, B, C, M, N, K, alpha, beta, handle);
            break;
        }
    }
    else if constexpr (std::is_same_v<T, __nv_bfloat16>){
        switch (config.type){
        case KernelType::HOPPER:
            switch (static_cast<HopperKernelVariant>(config.kernel_id)){
            case HopperKernelVariant::gemm_bf16_wgmma_tma:
                hopper::run_gemm_bf16_wgmma_tma(A, B, C, M, N, K, alpha, beta);
                break;
            case HopperKernelVariant::gemm_bf16_wgmma_tma_shapes:
                hopper::run_gemm_bf16_wgmma_tma_shapes(A, B, C, M, N, K, alpha, beta);
                break;
            case HopperKernelVariant::gemm_bf16_pc_pipeline:
                hopper::run_gemm_bf16_pc_pipeline(A, B, C, M, N, K, alpha, beta);
                break;
            default:
                throw std::invalid_argument("Unknown Hopper kernel ID");
            }
            break;

        case KernelType::CUBLAS:
            // Use run_gemm_cublas_bf16 for GENERAL kernels, run_gemm_cublas_bf16_h100 for HOPPER kernels
            if (comparison_type == KernelType::GENERAL) {
                cublas::run_gemm_cublas_bf16(A, B, C, M, N, K, alpha, beta, handle);
            } else {
                cublas::run_gemm_cublas_bf16_h100(A, B, C, M, N, K, alpha, beta, handle);
            }
            break;

        case KernelType::GENERAL:
            // Same variant ID, bf16 path calls the bf16 kernels
            switch (static_cast<GeneralKernelVariant>(config.kernel_id)){
            case GeneralKernelVariant::naive_sgemm:
                general::run_gemm_naive_bf16(A, B, C, M, N, K, alpha, beta);
                break;
            case GeneralKernelVariant::coalesced_sgemm:
                general::run_gemm_coalesced_bf16(A, B, C, M, N, K, alpha, beta);
                break;
            case GeneralKernelVariant::sgemm_tiled_shared:
                general::run_gemm_tiled_shared_bf16(A, B, C, M, N, K, alpha, beta);
                break;
            case GeneralKernelVariant::sgemm_1D_registertiling:
                general::run_gemm_1D_registertiling_bf16(A, B, C, M, N, K, alpha, beta);
                break;
            case GeneralKernelVariant::sgemm_2D_registertiling:
                general::run_gemm_2D_registertiling_bf16(A, B, C, M, N, K, alpha, beta);
                break;
            case GeneralKernelVariant::sgemm_vectorised:
                general::run_gemm_vectorised_bf16(A, B, C, M, N, K, alpha, beta);
                break;
            case GeneralKernelVariant::sgemm_warptiling:
                general::run_gemm_warptiling_bf16(A, B, C, M, N, K, alpha, beta);
                break;
            default:
                throw std::invalid_argument("Unknown general kernel ID");
            }
        }
    }
    else {
        static_assert(always_false<T>::value, "Unsupported element type for launchKernel");
    }
}
