/**
 * @file hopper_wgmma_utils.cuh
 *
 * @brief Host-side utilities for Tensor Memory Accelerator (TMA) on Hopper.
 */
#ifndef HOPPER_WGMMA_UTILS_CUH
#define HOPPER_WGMMA_UTILS_CUH

#include <cuda_runtime.h>
#include <cuda.h>
#include <type_traits>
#include <cuda_bf16.h> // for __nv_bfloat16
#include <cuda/barrier>

#include "utils.h"

// Alias for simplicity
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

/**
 * @brief Encodes a matrix descriptor value for WGMMA operations.
 *
 * Extracts and encodes the relevant bits from a 64-bit value for use in WGMMA matrix descriptors.
 * Masks the lower 18 bits and shifts right by 4 positions.
 *
 * @param x The 64-bit value to encode.
 * @return The encoded descriptor value.
 */
__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) {
    return ((x) & 0x3FFFF) >> 4;
}

/**
 * @brief Creates a WGMMA matrix descriptor for shared memory access.
 *
 * Constructs a 64-bit descriptor that encodes the layout and access pattern for a matrix
 * stored in shared memory. The descriptor specifies the base address, leading dimension,
 * stride dimension, and swizzle mode for WGMMA matrix multiply operations.
 *
 * @param ptr Pointer to the matrix data in shared memory.
 * @return A 64-bit WGMMA matrix descriptor with:
 *         - Bits [13:0]:  Encoded matrix start address
 *         - Bits [29:16]: Leading dimension byte offset (16 bytes)
 *         - Bits [45:32]: Stride dimension byte offset (1024 bytes)
 *         - Bits [62:63]: Swizzle mode (128B swizzle)
 */
__device__ uint64_t make_smem_desc(bf16* ptr) {
    uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    // Initialise an empty 64 bit descriptor
    uint64_t desc = 0x0000000000000000;
    // bitwise OR
    // sets bits [13:0] encoded matrix start address
    desc |= matrix_descriptor_encode(address);
    // sets bits [29:16] leading dimension byte offset
    desc |= matrix_descriptor_encode(static_cast<uint64_t>(16)) << 16;
    // sets bits [45: 32] stride dimension byte offset
    desc |= matrix_descriptor_encode(static_cast<uint64_t>(1024)) << 32;
    // sets bits [62: 63] swizzle mode
    desc |= 1llu << 62;
    return desc;
}

#endif
