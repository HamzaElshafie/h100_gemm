#pragma once

#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>


template <const uint BLOCK_SIZE>
__global__ void sgemm_coalesced(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, float alpha, float beta) {
        // flattened IDs remapping
        uint row = blockIdx.y * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
        uint column = blockIdx.x * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);

        if (row < M && column < K) {
            float cumulative_sum = 0.0f;
            for (int n = 0; n < N; n++) {
                cumulative_sum += A[row * N + n] * B[n * K + column];
            }
            C[row * K + column] = (alpha * cumulative_sum) + (beta * (C[row * K + column]));
        }
    }