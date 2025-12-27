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

template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma64(float d[4][8], bf16 *sA, bf16 *sB) {
    uint64_t desc_a = make_smem_desc(&sA[0]);
    uint64_t desc_b = make_smem_desc(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
        " %32,"
        " %33,"
        " %34, %35, %36, %37, %38;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
          "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]),
          "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]),
          "+f"(d[3][6]), "+f"(d[3][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

__device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ void warpgroup_wait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

#endif
