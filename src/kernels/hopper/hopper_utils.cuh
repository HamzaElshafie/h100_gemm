/**
 * @file hopper_utils.h
 *
 * @brief Utility functions for hopper-specific kernels
 */
#ifndef HOPPER_UTILS
#define HOPPER_UTILS

#include <iostream>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <type_traits>
#include <cuda_bf16.h> // for __nv_bfloat16

#include "utils.h"

// Alias for simplicity
using bf16 = __nv_bfloat16;

/**
 * @brief Creates a CUDA tensor map descriptor for TMA operations.
 *
 * This function encodes the layout and properties of a tensor in global memory into a CUtensorMap structure,
 * using the CUDA Driver API. The tensor map is required for Tensor Memory Accelerator (TMA) asynchronous memory operations
 * on Hopper.
 *
 * @tparam BlockMinorSize   The tile size in the minor (fastest-changing) dimension.
 * @tparam BlockMajorSize   The tile size in the major (slowest-changing) dimension.
 * @param tensorMap         Pointer to the CUtensorMap structure to be filled.
 * @param tensor_ptr        Pointer to the tensor data in global memory.
 * @param height            Number of rows in the tensor.
 * @param width             Number of columns in the tensor.
 */
template <const uint BlockMinorSize, const uint BlockMajorSize>
void create_tensor_map(CUtensorMap *tensorMap, bf16 *tensor_ptr, uint height, uint width)
{
    // Starting address of memory region described by tensor (casting to void
    // as the tensor map descriptor is type-agnostic.)
    void *gmem_address = static_cast<void *>(tensor_ptr);

    uint num_tiles_minor = CEIL_DIV(width, BlockMinorSize);
    uint num_tiles_major = CEIL_DIV(height, BlockMajorSize);

    // full size of the tensor in global memory (API expects the 5D supported
    // tensor ranks to be defined)
    uint64_t global_dim[5] = {
        static_cast<uint64_t>(BlockMinorSize * num_tiles_minor),
        static_cast<uint64_t>(BlockMajorSize * num_tiles_major),
        1, 1, 1};

    // Define the tensor strides (in bytes) along each of the tensor ranks dims - 1
    uint64_t global_strides[5] = {sizeof(bf16), sizeof(bf16) * BlockMinorSize * num_tiles_minor, 0, 0, 0};

    // Define the shape of the "box_size" -> the tile shapes a TMA ops will load
    uint32_t box_dim[5] = {
        static_cast<uint32_t>(BlockMinorSize),
        static_cast<uint32_t>(BlockMajorSize),
        1, 1, 1};

    uint32_t elem_strides[5] = {1, 1, 1, 1, 1};

    // Create tensor map
    CU_CHECK(cuTensorMapEncodeTiled(
        tensorMap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gmem_address,
        global_dim, global_strides + 1, box_dim, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

#endif
