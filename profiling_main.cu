/**
 * @file profiling_main.cu
 * @brief Standalone profiling program for gemm_bf16_pc_pipeline kernel
 * 
 * This file contains all necessary utilities, kernel code, and main program
 * for profiling the producer-consumer pipeline GEMM kernel.
 */

#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cuda/barrier>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <type_traits>
#include "utils.h"

// Alias for simplicity
using bf16 = __nv_bfloat16;
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

/**
 * @brief Creates a CUDA tensor map descriptor for TMA operations.
 */
template <const uint BlockMajorSize, const uint BlockMinorSize>
void create_tensor_map(CUtensorMap *tensor_map, bf16 *tensor_ptr, uint blocks_height, uint blocks_width) {
    void *gmem_address = static_cast<void *>(tensor_ptr);

    uint num_tiles_major = blocks_height;
    uint num_tiles_minor = blocks_width;

    uint64_t global_dim[5] = {
        static_cast<uint64_t>(BlockMinorSize * num_tiles_minor),
        static_cast<uint64_t>(BlockMajorSize * num_tiles_major),
        1, 1, 1};

    uint64_t global_strides[5] = {
        sizeof(bf16),
        sizeof(bf16) * BlockMinorSize * num_tiles_minor,
        0, 0, 0};

    uint32_t box_dim[5] = {
        static_cast<uint32_t>(BlockMinorSize),
        static_cast<uint32_t>(BlockMajorSize),
        1, 1, 1};

    uint32_t elem_strides[5] = {1, 1, 1, 1, 1};

    CU_CHECK(cuTensorMapEncodeTiled(
        tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gmem_address,
        global_dim, global_strides + 1, box_dim, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

/**
 * @brief Allocates device memory and creates a CUDA tensor map descriptor for TMA operations.
 */
template <const uint BlockMajorSize, const uint BlockMinorSize>
__host__ static inline CUtensorMap*
create_and_allocate_tensor_map(bf16 *tensor_ptr, uint blocks_height, uint blocks_width) {
    CUtensorMap *tensor_map;
    CUDA_CHECK(cudaMalloc((void **)&tensor_map, sizeof(CUtensorMap)));
    CUtensorMap tensor_map_host;
    create_tensor_map<BlockMajorSize, BlockMinorSize>(&tensor_map_host, tensor_ptr, blocks_height, blocks_width);
    CUDA_CHECK(cudaMemcpy(tensor_map, &tensor_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice));
    return tensor_map;
}


/**
 * @brief Encodes a matrix descriptor value for WGMMA operations.
 */
__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) {
    return ((x) & 0x3FFFF) >> 4;
}

/**
 * @brief Creates a WGMMA matrix descriptor for shared memory access.
 */
__device__ uint64_t make_smem_desc(bf16* ptr) {
    uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(address);
    desc |= matrix_descriptor_encode(static_cast<uint64_t>(16)) << 16;
    desc |= matrix_descriptor_encode(static_cast<uint64_t>(1024)) << 32;
    desc |= 1llu << 62;
    return desc;
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

template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma128(float d[8][8], bf16 *sharedA, bf16 *sharedB) {
    uint64_t desc_a = make_smem_desc(&sharedA[0]);
    uint64_t desc_b = make_smem_desc(&sharedB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " %66,    %67,  %68,  %69,  %70;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
          "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]),
          "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]),
          "+f"(d[3][6]), "+f"(d[3][7]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
          "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]), "+f"(d[5][0]), "+f"(d[5][1]),
          "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
          "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]),
          "+f"(d[6][6]), "+f"(d[6][7]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
          "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

/**
 * Compile-time dispatcher that selects the correct WGMMA instruction variant based on WGMMA_N.
 */
template <int WGMMA_N, int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ inline void wgmma(float d[WGMMA_N / 16][8], bf16 *sharedA, bf16 *sharedB){
    if constexpr (WGMMA_N == 128){wgmma128<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, sharedA, sharedB);}
}

template <int TILE_SIZE_M, int TILE_SIZE_K, int TILE_SIZE_N, int NUM_STAGES>
struct Smem {
    alignas(128) bf16 A[TILE_SIZE_M * TILE_SIZE_K * NUM_STAGES];
    alignas(128) bf16 B[TILE_SIZE_K * TILE_SIZE_N * NUM_STAGES];

    static constexpr int TILE_M_PAD = TILE_SIZE_M + 8;
    // Epilogue staging tile (padded)
    alignas(128) bf16 C_epi[TILE_M_PAD * TILE_SIZE_N];
};

template <const int TILE_SIZE_M, const int TILE_SIZE_K, const int TILE_SIZE_N,
          const int WGMMA_M, const int WGMMA_N, const int WGMMA_K, const int NUM_THREADS,
          const int NUM_STAGES = 5>
__global__ void __launch_bounds__(NUM_THREADS)
gemm_bf16_pc_pipeline(CUtensorMap* tensorMapA, CUtensorMap* tensorMapB, bf16* C,
    int M, int K, int N, float alpha, float beta) {
        static_assert(WGMMA_N == TILE_SIZE_N, "WGMMA_N must be == TILE_SIZE_N");
        static_assert(TILE_SIZE_M % WGMMA_M == 0, "TILE_SIZE_M must be divisible by WGMMA_M");
        static_assert(TILE_SIZE_K % WGMMA_K == 0, "TILE_SIZE_K must be divisible by WGMMA_K");
        static_assert(TILE_SIZE_N % WGMMA_N == 0, "TILE_SIZE_N must be divisible by WGMMA_N");
        static_assert(NUM_THREADS % 128 == 0, "NUM_THREADS must be divisible by warp group size (128)");
        static_assert(NUM_THREADS >= 256, "Need at least 2 warp groups (1 producer + 1 consumer)");

        // Allocate SMEM
        extern __shared__ __align__(128) uint8_t smem_raw[];
        Smem<TILE_SIZE_M, TILE_SIZE_K, TILE_SIZE_N, NUM_STAGES> &s =
            *reinterpret_cast<Smem<TILE_SIZE_M, TILE_SIZE_K, TILE_SIZE_N, NUM_STAGES>*>(smem_raw);

        constexpr int TILE_M_PAD = Smem<TILE_SIZE_M, TILE_SIZE_K, TILE_SIZE_N, NUM_STAGES>::TILE_M_PAD;

        constexpr int A_stage_size = TILE_SIZE_M * TILE_SIZE_K;
        constexpr int B_stage_size = TILE_SIZE_K * TILE_SIZE_N;

        // How many warp groups in the block?
        constexpr int num_warp_groups = NUM_THREADS / 128;
        constexpr int num_consumer_groups = num_warp_groups - 1; // only 1 producer

        int warp_group_idx = threadIdx.x / 128;
        bool is_producer = (warp_group_idx == 0);

        // How many M rows of the output tile each 'consumer' warp group is responsible for
        // @example: TILE_SIZE_M = 128, num_consumer_groups = 1 -> 128 rows; num_consumer_groups = 2 -> 64 rows each
        constexpr int rows_per_consumer_warp_group = TILE_SIZE_M / num_consumer_groups;

        // Consumer warp group index (0-indexed among consumers only)
        int consumer_warp_group_idx = is_producer ? -1 : (warp_group_idx - 1);

        const int num_blocks_k = CEIL_DIV(K, TILE_SIZE_K);
        int num_block_m = blockIdx.x / CEIL_DIV(N, TILE_SIZE_N);
        int num_block_n = blockIdx.x % CEIL_DIV(N, TILE_SIZE_N);

        #pragma nv_diag_suppress static_var_with_dynamic_init
        __shared__ barrier full[NUM_STAGES];  // Signals data is ready
        __shared__ barrier empty[NUM_STAGES]; // Signals slot is available

        if (threadIdx.x == 0) {
            for (int i = 0; i < NUM_STAGES; i++) {
                init(&full[i], num_consumer_groups * 128 + 1); // consumers + producer thread 0
                init(&empty[i], num_consumer_groups * 128 + 1);
            }
            cde::fence_proxy_async_shared_cta();
        }
        __syncthreads();

        if (is_producer) {
            // Producer warp group: Issues TMA loads
            if (threadIdx.x == 0) {
                // Fill the pipeline
                for (int stage = 0; stage < NUM_STAGES && stage < num_blocks_k; stage++) {
                    int block_k_iter = stage;
                    
                    // Wait for empty slot (initially all are empty, so this passes immediately)
                    empty[stage].wait(empty[stage].arrive());

                    // Get pointers for this stage in the flat arrays
                    bf16* A_stage = s.A + (stage * A_stage_size);
                    bf16* B_stage = s.B + (stage * B_stage_size);

                    // TMA loads for A and B
                    cde::cp_async_bulk_tensor_2d_global_to_shared(A_stage, tensorMapA, block_k_iter * TILE_SIZE_K, num_block_m * TILE_SIZE_M, full[stage]);
                    cde::cp_async_bulk_tensor_2d_global_to_shared(B_stage, tensorMapB, block_k_iter * TILE_SIZE_K, num_block_n * TILE_SIZE_N, full[stage]);

                    // Signal data is ready
                    barrier::arrival_token token = cuda::device::barrier_arrive_tx(full[stage], 1, A_stage_size * sizeof(bf16) + B_stage_size * sizeof(bf16));
                }

                // Main loop: Continue issuing loads
                for (int block_k_iter = NUM_STAGES; block_k_iter < num_blocks_k; block_k_iter++) {
                    int stage = block_k_iter % NUM_STAGES;
                    
                    // Wait for this stage to be empty before overwriting
                    empty[stage].wait(empty[stage].arrive());

                    // Get pointers for this stage in the flat arrays
                    bf16* A_stage = s.A + (stage * A_stage_size);
                    bf16* B_stage = s.B + (stage * B_stage_size);

                    // Issue next TMA loads
                    cde::cp_async_bulk_tensor_2d_global_to_shared(A_stage, tensorMapA, block_k_iter * TILE_SIZE_K, num_block_m * TILE_SIZE_M, full[stage]);
                    cde::cp_async_bulk_tensor_2d_global_to_shared(B_stage, tensorMapB, block_k_iter * TILE_SIZE_K, num_block_n * TILE_SIZE_N, full[stage]);

                    // Signal data is ready
                    barrier::arrival_token token = cuda::device::barrier_arrive_tx(full[stage], 1, A_stage_size * sizeof(bf16) + B_stage_size * sizeof(bf16));
                }
            }
            
        } else {
            // Consumer warp groups: Execute WGMMA compute
            // Accumulator registers - declared inside consumer branch only so
            // ptxas doesn't allocate them for the producer warp group
            float d[TILE_SIZE_M / WGMMA_M / num_consumer_groups][WGMMA_N / 16][8];
            memset(d, 0, sizeof(d));

            // Initially signal all empty slots are available
            for (int i = 0; i < NUM_STAGES; i++) {
                barrier::arrival_token token = empty[i].arrive();
            }

            // Main compute loop
            for (int block_k_iter = 0; block_k_iter < num_blocks_k; block_k_iter++) {
                int stage = block_k_iter % NUM_STAGES;
                
                // Get pointers for this stage in the flat arrays
                bf16* A_stage = s.A + (stage * A_stage_size);
                bf16* B_stage = s.B + (stage * B_stage_size);
                
                // Wait for data to be ready
                full[stage].arrive_and_wait();

                // Compute phase using WGMMA
                warpgroup_arrive();
                
                #pragma unroll
                for (int m_iter = 0; m_iter < rows_per_consumer_warp_group / WGMMA_M; m_iter++) {
                    bf16* sharedA_wgmma_tile_base = A_stage + ((consumer_warp_group_idx * rows_per_consumer_warp_group) + (m_iter * WGMMA_M)) * TILE_SIZE_K;
                    
                    #pragma unroll
                    for (int k_iter = 0; k_iter < TILE_SIZE_K / WGMMA_K; k_iter++) {
                        wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[m_iter], &sharedA_wgmma_tile_base[k_iter * WGMMA_K], &B_stage[k_iter * WGMMA_K]);
                    }
                }
                
                warpgroup_commit_batch();
                warpgroup_wait<0>();

                // Signal this slot is now empty and can be reused
                barrier::arrival_token empty_token = empty[stage].arrive();
            }

            int tid  = threadIdx.x % 128;
            int lane = tid % 32;
            int warp = tid / 32;
            uint32_t row = warp * 16 + lane / 4;

            // @note C is column-major
            bf16* block_C = C + (num_block_n * TILE_SIZE_N * M) + (num_block_m * TILE_SIZE_M);

            constexpr int TILE_M_PAD = TILE_SIZE_M + 8;
            #define IDX_GMEM(r, c) ((c) * M + (r))
            #define IDX_SMEM(r, c) ((c) * TILE_M_PAD + (r))

            // Phase 1: alpha-scaled accumulators -> shared staging tile
            for (int m_iter = 0; m_iter < rows_per_consumer_warp_group / WGMMA_M; m_iter++) {
                int row_tile_base_C = (consumer_warp_group_idx * rows_per_consumer_warp_group) + (m_iter * WGMMA_M);
                for (int w = 0; w < WGMMA_N / 16; w++) {
                    int col = 16 * w + 2 * (tid % 4);
                    s.C_epi[IDX_SMEM(row + row_tile_base_C, col)] = __float2bfloat16(alpha * d[m_iter][w][0]);
                    s.C_epi[IDX_SMEM(row + row_tile_base_C, col + 1)] = __float2bfloat16(alpha * d[m_iter][w][1]);
                    s.C_epi[IDX_SMEM(row + 8 + row_tile_base_C, col)] = __float2bfloat16(alpha * d[m_iter][w][2]);
                    s.C_epi[IDX_SMEM(row + 8 + row_tile_base_C, col + 1)] = __float2bfloat16(alpha * d[m_iter][w][3]);
                    s.C_epi[IDX_SMEM(row + row_tile_base_C, col + 8)] = __float2bfloat16(alpha * d[m_iter][w][4]);
                    s.C_epi[IDX_SMEM(row + row_tile_base_C, col + 9)] = __float2bfloat16(alpha * d[m_iter][w][5]);
                    s.C_epi[IDX_SMEM(row + 8 + row_tile_base_C, col + 8)] = __float2bfloat16(alpha * d[m_iter][w][6]);
                    s.C_epi[IDX_SMEM(row + 8 + row_tile_base_C, col + 9)] = __float2bfloat16(alpha * d[m_iter][w][7]);
                }
            }
            __syncthreads();

            // Phase 2: coalesced write to GMEM (alpha*D + beta*C)
            int row4_in_group = lane * 4;
            int group_base_row = consumer_warp_group_idx * rows_per_consumer_warp_group;
            if (row4_in_group < rows_per_consumer_warp_group) {
                int r0 = group_base_row + row4_in_group;
                for (int c = warp; c < TILE_SIZE_N; c += 4) {
                    block_C[IDX_GMEM(r0 + 0, c)] = __float2bfloat16(__bfloat162float(s.C_epi[IDX_SMEM(r0 + 0, c)]) + beta * __bfloat162float(block_C[IDX_GMEM(r0 + 0, c)]));
                    block_C[IDX_GMEM(r0 + 1, c)] = __float2bfloat16(__bfloat162float(s.C_epi[IDX_SMEM(r0 + 1, c)]) + beta * __bfloat162float(block_C[IDX_GMEM(r0 + 1, c)]));
                    block_C[IDX_GMEM(r0 + 2, c)] = __float2bfloat16(__bfloat162float(s.C_epi[IDX_SMEM(r0 + 2, c)]) + beta * __bfloat162float(block_C[IDX_GMEM(r0 + 2, c)]));
                    block_C[IDX_GMEM(r0 + 3, c)] = __float2bfloat16(__bfloat162float(s.C_epi[IDX_SMEM(r0 + 3, c)]) + beta * __bfloat162float(block_C[IDX_GMEM(r0 + 3, c)]));
                }
            }
            #undef IDX_GMEM
            #undef IDX_SMEM
        }
}

/**
 * @brief Main entry point for the profiling program (gemm_bf16_pc_pipeline only).
 */
int main(int argc, char** argv) {
    ResourceManager<bf16> resources;

    // Fixed size for profiling
    const int size = 8192;
    const float alpha = 0.5f;
    const float beta = 3.0f;
    
    const size_t mem_size = size * size * sizeof(bf16);

    std::cout << "Profiling gemm_bf16_pc_pipeline kernel with matrix size " << size << "x" << size << std::endl;

    // Allocate host memory
    bf16* A_host = (bf16*)malloc(mem_size);
    bf16* B_host = (bf16*)malloc(mem_size);
    bf16* C_host = (bf16*)malloc(mem_size);

    // Register host memory with resource manager
    resources.add_host_ptr(A_host);
    resources.add_host_ptr(B_host);
    resources.add_host_ptr(C_host);

    if (!A_host || !B_host || !C_host) {
        std::cerr << "Host memory allocation failed" << std::endl;
        return -1;
    }

    // Initialise matrices
    bf16* matrices[] = {A_host, B_host, C_host};
    initialiseArrays<bf16>(matrices, 3, size * size, -100.0f, 100.0f, 0);

    // Allocate device memory
    bf16* A_device;
    bf16* B_device;
    bf16* C_device;

    CUDA_CHECK(cudaMalloc((void**)&A_device, mem_size));
    CUDA_CHECK(cudaMalloc((void**)&B_device, mem_size));
    CUDA_CHECK(cudaMalloc((void**)&C_device, mem_size));

    resources.add_device_ptr(A_device);
    resources.add_device_ptr(B_device);
    resources.add_device_ptr(C_device);

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(A_device, A_host, mem_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_device, B_host, mem_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C_device, C_host, mem_size, cudaMemcpyHostToDevice));

    std::cout << "Memory allocation and data transfer completed" << std::endl;

    // Kernel launch configuration
    constexpr int TILE_SIZE_M = 128;
    constexpr int TILE_SIZE_N = 128;
    constexpr int TILE_SIZE_K = 64;
    constexpr int NUM_THREADS = 128 * 2;
    constexpr int NUM_STAGES = 5;
    constexpr int WGMMA_M = 64;
    constexpr int WGMMA_K = 16;
    constexpr int WGMMA_N = 128;

    // Create tensor maps
    CUtensorMap* d_tma_map_A = create_and_allocate_tensor_map<TILE_SIZE_M, TILE_SIZE_K>(
        A_device, CEIL_DIV(size, TILE_SIZE_M), CEIL_DIV(size, TILE_SIZE_K));
    CUtensorMap* d_tma_map_B = create_and_allocate_tensor_map<TILE_SIZE_N, TILE_SIZE_K>(
        B_device, CEIL_DIV(size, TILE_SIZE_N), CEIL_DIV(size, TILE_SIZE_K));

    resources.add_device_ptr(reinterpret_cast<bf16*>(d_tma_map_A));
    resources.add_device_ptr(reinterpret_cast<bf16*>(d_tma_map_B));

    // Configure kernel
    auto* kernel = gemm_bf16_pc_pipeline<
        TILE_SIZE_M, TILE_SIZE_K, TILE_SIZE_N,
        WGMMA_M, WGMMA_N, WGMMA_K, NUM_THREADS, NUM_STAGES>;
    size_t sMemSize = sizeof(Smem<TILE_SIZE_M, TILE_SIZE_K, TILE_SIZE_N, NUM_STAGES>);
    CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    // Grid dimensions
    dim3 grid((size / TILE_SIZE_M) * (size / TILE_SIZE_N));
    dim3 block(NUM_THREADS);

    // Warm-up launches
    std::cout << "Running warm-up launches..." << std::endl;
    for (int i = 0; i < 2; ++i) {
        kernel<<<grid, block, sMemSize>>>(d_tma_map_A, d_tma_map_B, C_device, size, size, size, alpha, beta);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Main kernel launch for profiling
    std::cout << "Running main kernel for profiling..." << std::endl;
    kernel<<<grid, block, sMemSize>>>(d_tma_map_A, d_tma_map_B, C_device, size, size, size, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Kernel launch completed successfully" << std::endl;

    return 0;  // ResourceManager destructor will handle cleanups
}
