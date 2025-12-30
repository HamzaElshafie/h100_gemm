/**
 * @file hopper_tma_utils.h
 *
 * @brief Host-side utilities for Tensor Memory Accelerator (TMA) on Hopper.
 */
#ifndef HOPPER_TMA_UTILS_H
#define HOPPER_TMA_UTILS_H

#include <cuda_runtime.h>
#include <cuda.h>
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
 * @cite https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
 *
 * @tparam BlockMinorSize   The tile size in the minor (fastest-changing) dimension.
 * @tparam BlockMajorSize   The tile size in the major (slowest-changing) dimension.
 * @param tensorMap         Pointer to the CUtensorMap structure to be filled.
 * @param tensor_ptr        Pointer to the tensor data in global memory.
 * @param height            Number of rows in the tensor.
 * @param width             Number of columns in the tensor.
 */
template <const uint BlockMajorSize, const uint BlockMinorSize>
void create_tensor_map(CUtensorMap *tensor_map, bf16 *tensor_ptr, uint blocks_height, uint blocks_width) {
    // Starting address of memory region described by tensor (casting to void
    // as the tensor map descriptor is type-agnostic.)
    void *gmem_address = static_cast<void *>(tensor_ptr);

    uint num_tiles_major = blocks_height;
    uint num_tiles_minor = blocks_width;

    // full size of the tensor in global memory (API expects the 5D supported
    // tensor ranks to be defined)
    uint64_t global_dim[5] = {
        static_cast<uint64_t>(BlockMinorSize * num_tiles_minor),
        static_cast<uint64_t>(BlockMajorSize * num_tiles_major),
        1, 1, 1};

    // Define the tensor strides (in bytes) along each of the tensor ranks dims - 1
    uint64_t global_strides[5] = {
        sizeof(bf16),
        sizeof(bf16) * BlockMinorSize * num_tiles_minor,
        0, 0, 0};

    // Define the shape of the "box_size" -> the tile shapes a TMA ops will load
    uint32_t box_dim[5] = {
        static_cast<uint32_t>(BlockMinorSize),
        static_cast<uint32_t>(BlockMajorSize),
        1, 1, 1};

    uint32_t elem_strides[5] = {1, 1, 1, 1, 1};

    // Create tensor map
    CU_CHECK(cuTensorMapEncodeTiled(
        tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gmem_address,
        global_dim, global_strides + 1, box_dim, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

/**
 * @brief Allocates device memory and creates a CUDA tensor map descriptor for TMA operations.
 *
 * This function allocates device memory for a CUtensorMap descriptor, creates the tensor map on the host
 * using the provided tensor pointer and dimensions, and copies the descriptor to the device. It internally
 * calls create_tensor_map() to perform the host-side descriptor creation. The resulting device pointer can be
 * used in kernels that require TMA tensor maps on Hopper architectures.
 *
 * @tparam BlockMinorSize   The tile size in the minor (fastest-changing) dimension.
 * @tparam BlockMajorSize   The tile size in the major (slowest-changing) dimension.
 * @param tensor_ptr        Pointer to the tensor data in global memory.
 * @param height            Number of rows in the tensor.
 * @param width             Number of columns in the tensor.
 * @return                  Device pointer to the allocated and initialized CUtensorMap descriptor.
 */
template <const uint BlockMajorSize, const uint BlockMinorSize>
__host__ static inline CUtensorMap *
create_and_allocate_tensor_map(bf16 *tensor_ptr, uint height, uint width) {
    CUtensorMap *tensor_map;
    // Allocate device memory for the tensor map descriptor.
    CUDA_CHECK(cudaMalloc((void **)&tensor_map, sizeof(CUtensorMap)));
    // Register the tensorMap in our device memory pointers
    // resources.add_device_ptr(tensor_map);
    // Create on host
    CUtensorMap tensor_map_host;
    create_tensor_map<BlockMajorSize, BlockMinorSize>(&tensor_map_host, tensor_ptr, height, width);
    // Copy descriptor to device
    CUDA_CHECK(cudaMemcpy(tensor_map, &tensor_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice));
    return tensor_map;
}

#endif