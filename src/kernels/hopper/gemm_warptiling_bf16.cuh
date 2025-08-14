#pragma once

#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>

#define WARPSIZE 32

template <const uint TILE_SIZE_M, const uint TILE_SIZE_N, const uint TILE_SIZE_K,
        const uint WARP_TILE_M, const uint WARP_TILE_K, const uint WARP_STEPS_K,
        const uint ROWS_PER_THREAD, const uint COLS_PER_THREAD, const uint NUM_THREADS,
        bool PAD_SMEM_A = false, bool PAD_SMEM_B = false>
__global__ void __launch_bounds__(NUM_THREADS) 
    gemm_warptiling_bf16(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C,
        int M, int N, int K, float alpha, float beta) {

        static_assert(TILE_SIZE_K % WARP_TILE_K == 0, "TILE_SIZE_K must tile by WARP_TILE_K");

        // Allocate shared memory
        __shared__ __nv_bfloat16 sharedA[TILE_SIZE_M * TILE_SIZE_N];
        __shared__ __nv_bfloat16 sharedB[TILE_SIZE_N * TILE_SIZE_K];

        // Identify the tile of C this thread block is responsible for
        const uint block_row = blockIdx.y;
        const uint block_column = blockIdx.x;

        constexpr uint WARP_STEPS_M = (WARP_TILE_M * WARP_TILE_K) / (WARPSIZE * (ROWS_PER_THREAD * COLS_PER_THREAD)); // 1
        // constexpr uint WARP_STEPS_M = WARP_TILE_M / ROWS_PER_THREAD; Try this
        // Warp subtile is WARP_SUB_M x WARP_SUB_K
        constexpr uint WARP_SUB_M = WARP_TILE_M / WARP_STEPS_M; // 64 / 1 = 64
        constexpr uint WARP_SUB_K = WARP_TILE_K / WARP_STEPS_K; // 64 / 4 = 16

        // Identify the warp tile position
        const uint warp_idx = threadIdx.x / WARPSIZE; // threads 0,..,31 --> warp_idx = 0,  threads 32,..,63 --> warp_idx = 1 etc
        const uint warps_per_row = TILE_SIZE_K / WARP_TILE_K; // 2
        const uint warp_row = warp_idx / warps_per_row; // (0, 1). Given our launch configs ofc
        const uint warp_col = warp_idx % warps_per_row; // (0, 1)

        // Identify the thread position within the warp tile
        uint lane = threadIdx.x % WARPSIZE;
        const uint threads_per_subtile_row = WARP_SUB_K / COLS_PER_THREAD; // 16/4 = 4 threads per row
        const uint thread_row_in_sub = lane / threads_per_subtile_row; // Eg. 6/4 = 1.5 = 1. Row 1 
        const uint thread_col_in_sub = lane % threads_per_subtile_row; // Eg. 5%4 = 1. Col 1

        const uint ty = thread_row_in_sub;
        const uint tx = thread_col_in_sub;

        // Move pointers from A[0], B[0] and C[0] to the starting positions of the tile
        A += block_row * TILE_SIZE_M * N;                                  // Move pointer (block_row * TILE_SIZE_M) rows down
        B += block_column * TILE_SIZE_K;                                   // Move pointer (block_column * TILE_SIZE_K) columns to the right
        C += (block_row * TILE_SIZE_M * K) + (block_column * TILE_SIZE_K); // Move pointer (block_row * TILE_SIZE_M * K) rows down then (block_column * TILE_SIZE_K) columns to the right
        
        const uint VEC_CHUNKS_N = TILE_SIZE_N / 4; // 16 / 4 = 4
        const uint VEC_CHUNKS_K = TILE_SIZE_K / 4; // 128 / 4 = 32

        // Map each thread to one 4-float chunk that it will load. We will have to offset as we have fewer threads than elements to cover (offset by 32)
        const uint smem_ty_A = threadIdx.x / VEC_CHUNKS_N; // --> 0, ..., 31
        const uint smem_tx_A = threadIdx.x % VEC_CHUNKS_N; // --> 0, 1, 2, 3
        // If we give each thread a vector load of 4 elements along TILE_SIZE_N, how many different rows of sharedA can we cover in one pass through all the threads?
        const uint strideA = (NUM_THREADS * 4) / TILE_SIZE_N; // 32 rows per pass

        const uint smem_ty_B = threadIdx.x / VEC_CHUNKS_K; // --> 0, 1, 2, 3
        const uint smem_tx_B = threadIdx.x % VEC_CHUNKS_K; // --> 0, ...., 31
        // If we give each thread a vector load of 4 elements along TILE_SIZE_K, how many different rows of sharedB can we cover in one pass through all the threads?
        const uint strideB = (NUM_THREADS * 4) / TILE_SIZE_K; // 4 rows per pass

       float thread_results[WARP_STEPS_M * ROWS_PER_THREAD * WARP_STEPS_K * COLS_PER_THREAD] = {0.0f};
       __nv_bfloat16 reg_m[WARP_STEPS_M * ROWS_PER_THREAD]; // In our case 1 x 8
       __nv_bfloat16 reg_k[WARP_STEPS_K * COLS_PER_THREAD]; // 4 x 4 = 16

       const uint num_tiles = CEIL_DIV(N, TILE_SIZE_N);

       // Outer loop iterate over tiles
       for (int t = 0; t < num_tiles; t++) {
        // Populate smem using vectorised loads (We use offsets and is transposed)
        for (int load_offset = 0; load_offset < TILE_SIZE_M; load_offset += strideA) { // 0, 32, 64, 96
            const float2 v = reinterpret_cast<const float2*>(&A[(smem_ty_A + load_offset) * N + smem_tx_A * 4])[0];
            __nv_bfloat16 tempA[4];
            memcpy(&tempA[0], &v, sizeof(__nv_bfloat16) * 4);

            // Transpose A (instead of 128x16 previously for ex, now it will be 16x128)
            sharedA[(smem_tx_A * 4 + 0) * TILE_SIZE_M + (smem_ty_A + load_offset)] = tempA[0];
            sharedA[(smem_tx_A * 4 + 1) * TILE_SIZE_M + (smem_ty_A + load_offset)] = tempA[1];
            sharedA[(smem_tx_A * 4 + 2) * TILE_SIZE_M + (smem_ty_A + load_offset)] = tempA[2];
            sharedA[(smem_tx_A * 4 + 3) * TILE_SIZE_M + (smem_ty_A + load_offset)] = tempA[3];
        }

        // Load from as B as well but without transposing
        for (int load_offset = 0; load_offset < TILE_SIZE_N; load_offset += strideB) { // 0, 4, 8, 12
            reinterpret_cast<float2*>(&sharedB[(smem_ty_B + load_offset) * TILE_SIZE_K + smem_tx_B * 4])[0] = 
                reinterpret_cast<const float2*>(&B[(smem_ty_B + load_offset) * K + smem_tx_B * 4])[0];
        }

        __syncthreads();

        // Iterate over the shared dimension of the SMEM tiles
        for (int i = 0; i < TILE_SIZE_N; i++) {
            // Load slice at current i iteration in sharedA's register
            for (int wSubRow = 0; wSubRow < WARP_STEPS_M; wSubRow++) {
                uint base_row = (warp_row * WARP_TILE_M) + (wSubRow * WARP_SUB_M) + (ty * ROWS_PER_THREAD);

                // Each thread loads ROWS_PER_THREAD into the register
                #pragma unroll
                for (int row = 0; row < ROWS_PER_THREAD; row+=4) {
                    const float2 va = reinterpret_cast<float2*>(
                        &sharedA[i * TILE_SIZE_M + base_row + row])[0];
                        
                    __nv_bfloat16 t4[4];
                    memcpy(&t4[0], &va, sizeof(__nv_bfloat16) * 4);

                    reg_m[wSubRow * ROWS_PER_THREAD + row + 0] = t4[0];
                    reg_m[wSubRow * ROWS_PER_THREAD + row + 1] = t4[1];
                    reg_m[wSubRow * ROWS_PER_THREAD + row + 2] = t4[2];
                    reg_m[wSubRow * ROWS_PER_THREAD + row + 3] = t4[3];
                }

            for (int wSubCol = 0; wSubCol < WARP_STEPS_K; wSubCol++) {
                uint col_base = (warp_col * WARP_TILE_K) + (wSubCol * WARP_SUB_K) + (tx * COLS_PER_THREAD);

                // Each thread loads COLS_PER_THREAD into the register x 4 times in our case since WARP_STEPS_K = 4
                #pragma unroll
                for (int col = 0; col < COLS_PER_THREAD; col+=4) {
                    const float2 vb = reinterpret_cast<float2*>(&sharedB[i * TILE_SIZE_K  + col_base + col])[0];

                    __nv_bfloat16 t4[4];
                    memcpy(&t4[0], &vb, sizeof(__nv_bfloat16) * 4);

                    reg_k[wSubCol * COLS_PER_THREAD + col + 0] = t4[0];
                    reg_k[wSubCol * COLS_PER_THREAD + col + 1] = t4[1];
                    reg_k[wSubCol * COLS_PER_THREAD + col + 2] = t4[2];
                    reg_k[wSubCol * COLS_PER_THREAD + col + 3] = t4[3];
                }
            }

            // Compute outer product of ROWS_PER_THREAD & COLS_PER_THREAD producing (8 * 1) x (4 x 4) = 8 x 16
            for (int wSubRow = 0; wSubRow < WARP_STEPS_M; wSubRow++) {
                for (int wSubCol = 0; wSubCol < WARP_STEPS_K; wSubCol++) {
                    #pragma unroll
                    for (int im = 0; im < ROWS_PER_THREAD; im++) {
                        // Fixate one value from reg_m
                        float fixed_temp =  __bfloat162float(reg_m[wSubRow * ROWS_PER_THREAD + im]);
                        #pragma unroll
                        for (int ik = 0; ik < COLS_PER_THREAD; ik++) {
                            float out = fixed_temp * __bfloat162float(reg_k[wSubCol * COLS_PER_THREAD + ik]);
                            // Flatten (row, col) into thread_results
                            // row_index   = wSubRow * ROWS_PER_THREAD + im
                            // col_index   = wSubCol * COLS_PER_THREAD + ik
                            // row_stride  = WARP_STEPS_K * COLS_PER_THREAD
                            int out_idx = (wSubRow * ROWS_PER_THREAD + im) * (WARP_STEPS_K * COLS_PER_THREAD) + (wSubCol * COLS_PER_THREAD + ik);
                            thread_results[out_idx] += out;
                        }
                    } 
                }
            }
            }
        }
        __syncthreads();

        A += TILE_SIZE_N; // Move right
        B += TILE_SIZE_N * K; // Move down     
       }

       // Write results of the thread back to C
       for (int wSubRow = 0; wSubRow < WARP_STEPS_M; wSubRow++) {
        for (int wSubCol = 0; wSubCol < WARP_STEPS_K; wSubCol++) {
            // Move C's pointer to the start of warp subtile
            C += (warp_row * WARP_TILE_M + wSubRow * WARP_SUB_M) * K + (warp_col * WARP_TILE_K + wSubCol * WARP_SUB_K);

            #pragma unroll
            for (int im = 0; im < ROWS_PER_THREAD; im++) {
                // Calculate global row
                int c_row = ty * ROWS_PER_THREAD + im; // Ex. ty(3). Row should be 3 * 8 + im
                #pragma unroll
                for (int ik = 0; ik < COLS_PER_THREAD; ik+=4) {
                    int c_col = tx * COLS_PER_THREAD + ik; // Ex. tx(2). Row should be 2 * 4 + im

                    uint idx_out = (wSubRow * ROWS_PER_THREAD + im) * (WARP_STEPS_K * COLS_PER_THREAD) + (wSubCol * COLS_PER_THREAD + ik);

                    // Store in a temporary array results of the current chunk across COLS_PER_THREAD
                    float temp_out[4];
                    temp_out[0] = thread_results[idx_out + 0];
                    temp_out[1] = thread_results[idx_out + 1];
                    temp_out[2] = thread_results[idx_out + 2];
                    temp_out[3] = thread_results[idx_out + 3];

                    __nv_bfloat16 out[4];

                    // One float2 packs four contiguous bf16 values from C
                    float2 vc = reinterpret_cast<float2*>(&C[c_row * K + c_col])[0]; 

                    // Reinterpret the 64 bits as four bf16 scalars
                    __nv_bfloat16 tempC[4];
                    memcpy(&tempC[0], &vc, sizeof(__nv_bfloat16) * 4);

                    // Convert to fp32 for computation
                    float tempC_fp32[4];
                    tempC_fp32[0] = __bfloat162float(tempC[0]);
                    tempC_fp32[1] = __bfloat162float(tempC[1]);
                    tempC_fp32[2] = __bfloat162float(tempC[2]);
                    tempC_fp32[3] = __bfloat162float(tempC[3]);

                    // Compute & convert back to bf16
                    out[0] = __float2bfloat16_rn(alpha * temp_out[0] + beta * tempC_fp32[0]);
                    out[1] = __float2bfloat16_rn(alpha * temp_out[1] + beta * tempC_fp32[1]);
                    out[2] = __float2bfloat16_rn(alpha * temp_out[2] + beta * tempC_fp32[2]);
                    out[3] = __float2bfloat16_rn(alpha * temp_out[3] + beta * tempC_fp32[3]);

                    float2 vo;
                    memcpy(&vo, &out[0], sizeof(__nv_bfloat16) * 4);
                    reinterpret_cast<float2*>(&C[c_row * K + c_col])[0] = vo;
                }
            }
        }
       }
    }