#pragma once

#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include "utils.h"
#include "hopper_wgmma_utils.cuh"
#include "hopper_tma_utils.h"

// Alias for simplicity
using bf16 = __nv_bfloat16;
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

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
            constexpr int num_regs_producer = (num_consumer_groups <= 2 ? 24 : 32);
            warpgroup_reg_dealloc<num_regs_producer>();
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
            constexpr int num_regs_consumer = (num_consumer_groups == 1 ? 256 : (num_consumer_groups == 2 ? 240 : 160));
            warpgroup_reg_alloc<num_regs_consumer>();
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