#pragma once

#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>

template <const uint BLOCKSIZE>
__global__ void sgemm_tiled_shared(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, float alpha, float beta) {
        // Allocate shared memory
        __shared__ float sharedA[BLOCKSIZE * BLOCKSIZE];
        __shared__ float sharedB[BLOCKSIZE * BLOCKSIZE];

        // Identify the tile of C this thread block is responsible for (We assume tiles are same size as block)
        const uint block_row = blockIdx.y;
        const uint block_column = blockIdx.x;

        // Calculate position of thread within tile (Remapping from 1-D to 2-D)
        const uint ty = threadIdx.x / BLOCKSIZE;
        const uint tx = threadIdx.x % BLOCKSIZE;
        
    }