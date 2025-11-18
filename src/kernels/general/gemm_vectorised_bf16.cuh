#pragma once

#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

template <const uint TILE_SIZE_M, const uint TILE_SIZE_N, const uint TILE_SIZE_K, const uint ROWS_PER_THREAD, const uint COLS_PER_THREAD>
__global__ void gemm_vectorised_bf16(const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K, float alpha, float beta) {
    // Allocate shared memory
    __shared__ __nv_bfloat16 sharedA[TILE_SIZE_M * TILE_SIZE_N];
    __shared__ __nv_bfloat16 sharedB[TILE_SIZE_N * TILE_SIZE_K];

    // Identify the tile of C this thread block is responsible for
    const uint block_row = blockIdx.y;
    const uint block_column = blockIdx.x;

    // Calculate position of thread within tile (Remapping from 1-D to 2-D) Note --> Each thread is a grid in itself hanlding ROWS_PER_THREAD x COLS_PER_THREAD
    const uint ty = threadIdx.x / (TILE_SIZE_K / COLS_PER_THREAD); // 0, ..., 15
    const uint tx = threadIdx.x % (TILE_SIZE_K / COLS_PER_THREAD); // 0, ..., 15

    // Move pointers from A[0], B[0] and C[0] to the starting positions of the tile
    A += block_row * TILE_SIZE_M * N;                                  // Move pointer (block_row * TILE_SIZE_M) rows down
    B += block_column * TILE_SIZE_K;                                   // Move pointer (block_column * TILE_SIZE_K) columns to the right
    C += (block_row * TILE_SIZE_M * K) + (block_column * TILE_SIZE_K); // Move pointer (block_row * TILE_SIZE_M * K) rows down then (block_column * TILE_SIZE_K) columns to the right

    // Map each thread to one 4-float chunk that it will load.
    const uint smem_ty_A = threadIdx.x / (TILE_SIZE_N / 4); // --> 0, ..., 127
    const uint smem_tx_A = threadIdx.x % (TILE_SIZE_N / 4); // --> 0, 1

    const uint smem_ty_B = threadIdx.x / (TILE_SIZE_K / 4); // --> 0, ..., 7
    const uint smem_tx_B = threadIdx.x % (TILE_SIZE_K / 4); // --> 0, ..., 31

    // Calculate how many tiles we have
    const uint num_tiles = CEIL_DIV(N, TILE_SIZE_N);
    float thread_results[ROWS_PER_THREAD * COLS_PER_THREAD] = {0.0f};
    __nv_bfloat16 reg_m[ROWS_PER_THREAD] = {};
    __nv_bfloat16 reg_k[COLS_PER_THREAD] = {};

    // Outer loop iterate over tiles
    for (int t = 0; t < num_tiles; t++) {
        // Populate smem using vector loads
        __nv_bfloat16 tempA[4];
        const float2 vA = reinterpret_cast<const float2*>(&A[smem_ty_A * N + smem_tx_A*4])[0]; 
        memcpy(&tempA[0], &vA, sizeof(__nv_bfloat16) * 4);
        // Transpose A (instead of 128x8 previously for ex, now it will be 8x128)
        sharedA[(smem_tx_A * 4 + 0) * TILE_SIZE_M + smem_ty_A] = tempA[0];
        sharedA[(smem_tx_A * 4 + 1) * TILE_SIZE_M + smem_ty_A] = tempA[1];
        sharedA[(smem_tx_A * 4 + 2) * TILE_SIZE_M + smem_ty_A] = tempA[2];
        sharedA[(smem_tx_A * 4 + 3) * TILE_SIZE_M + smem_ty_A] = tempA[3];

        __nv_bfloat16 tempB4[4];
        const float2 vB = reinterpret_cast<const float2*>(&B[smem_ty_B * K + smem_tx_B*4])[0];
        memcpy(&tempB4[0], &vB, sizeof(__nv_bfloat16) * 4);
        reinterpret_cast<float2*>(&sharedB[smem_ty_B * TILE_SIZE_K + smem_tx_B*4])[0] = *reinterpret_cast<float2*>(&tempB4[0]);

        __syncthreads();

        // Outer loop over shared dimension N
        for (int i = 0; i < TILE_SIZE_N; i++) {
            // Load into registers one "col" (its acc row now) from sharedA and one row from sharedB
            // We can actually also use vectorised loads here to cut smem load instructions by 4x.
            for (int row = 0; row < ROWS_PER_THREAD; row+=4) {
                uint global_smem_row_idx = ty * ROWS_PER_THREAD + row;
                // i will be the same for the whole "column" although since its transposed we are accessing same row. 
                // Notice how we also skip rows by TILE_SIZE_M now
                const float2 v0 = reinterpret_cast<float2*>(&sharedA[i * TILE_SIZE_M + global_smem_row_idx])[0]; // ld.shared.v4 via two fp32
                __nv_bfloat16 t0[4];
                memcpy(&t0[0], &v0, sizeof(__nv_bfloat16) * 4);
                reg_m[row + 0] = t0[0];
                reg_m[row + 1] = t0[1];
                reg_m[row + 2] = t0[2];
                reg_m[row + 3] = t0[3];
            }
            for (int col = 0; col < COLS_PER_THREAD; col+=4) {
                // We can do same vectorised loads 
                uint global_smem_col_idx = tx * COLS_PER_THREAD + col;
                const float2 v1 = reinterpret_cast<float2*>(&sharedB[i * TILE_SIZE_K + global_smem_col_idx])[0];
                __nv_bfloat16 t1[4];
                memcpy(&t1[0], &v1, sizeof(__nv_bfloat16) * 4);
                reg_k[col + 0] = t1[0];
                reg_k[col + 1] = t1[1];
                reg_k[col + 2] = t1[2];
                reg_k[col + 3] = t1[3];
            }
            
            // Calculate outer product between reg_m and reg_k to produce the partial results matrix of the thread 
            for (uint m = 0; m < ROWS_PER_THREAD; m++) {
                float am = __bfloat162float(reg_m[m]);
                for (uint k = 0; k < COLS_PER_THREAD; k++) {
                    thread_results[m * COLS_PER_THREAD + k] += am * __bfloat162float(reg_k[k]); // --> (ROWS_PER_THREAD x COLS_PER_THREAD) matrix
                }
            }
        }
        __syncthreads();

        A += TILE_SIZE_N; // Move right
        B += TILE_SIZE_N * K; // Move down
    }
    // Write results of the thread back to C
    for (uint row = 0; row < ROWS_PER_THREAD; row++) {
        // handle COLS_PER_THREAD in chunks of 4
        for (uint col = 0; col < COLS_PER_THREAD; col += 4) {
            uint global_row_idx = ty * ROWS_PER_THREAD + row;
            uint global_col_idx = tx * COLS_PER_THREAD + col;

            float temp_out[4];
            temp_out[0] = thread_results[row * COLS_PER_THREAD + col + 0];
            temp_out[1] = thread_results[row * COLS_PER_THREAD + col + 1];
            temp_out[2] = thread_results[row * COLS_PER_THREAD + col + 2];
            temp_out[3] = thread_results[row * COLS_PER_THREAD + col + 3];

            float2 vc = reinterpret_cast<float2*>(&C[global_row_idx * K + global_col_idx])[0];
            __nv_bfloat16 tempC[4];
            memcpy(&tempC[0], &vc, sizeof(__nv_bfloat16) * 4);
            float tempC_fp32[4];
            tempC_fp32[0] = __bfloat162float(tempC[0]);
            tempC_fp32[1] = __bfloat162float(tempC[1]);
            tempC_fp32[2] = __bfloat162float(tempC[2]);
            tempC_fp32[3] = __bfloat162float(tempC[3]);

            __nv_bfloat16 out[4];
            out[0] = __float2bfloat16_rn(alpha * temp_out[0] + beta * tempC_fp32[0]);
            out[1] = __float2bfloat16_rn(alpha * temp_out[1] + beta * tempC_fp32[1]);
            out[2] = __float2bfloat16_rn(alpha * temp_out[2] + beta * tempC_fp32[2]);
            out[3] = __float2bfloat16_rn(alpha * temp_out[3] + beta * tempC_fp32[3]);

            float2 vo;
            memcpy(&vo, &out[0], sizeof(__nv_bfloat16) * 4);
            reinterpret_cast<float2*>(&C[global_row_idx * K + global_col_idx])[0] = vo;
        }
    }
}


