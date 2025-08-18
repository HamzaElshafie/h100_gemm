#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

#include "utils.h"

#define WARPSIZE 32

template <const uint TILE_SIZE_M, const uint TILE_SIZE_N, const uint TILE_SIZE_K,
          const uint WARP_TILE_M, const uint WARP_TILE_K, const uint WARP_STEPS_K,
          const uint ROWS_PER_THREAD, const uint COLS_PER_THREAD, const uint NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
sgemm_warptiling(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                 int M, int N, int K, float alpha, float beta) {

    // Allocate shared memory. Use padded leading strides that keep float4 alignment
    constexpr uint STRIDE_A = (TILE_SIZE_M % 32u == 0u) ? (TILE_SIZE_M + 4u) : TILE_SIZE_M;
    constexpr uint STRIDE_B = (TILE_SIZE_K % 32u == 0u) ? (TILE_SIZE_K + 4u) : TILE_SIZE_K;
    static_assert((STRIDE_A % 4u) == 0u, "STRIDE_A must keep float4 alignment");
    static_assert((STRIDE_B % 4u) == 0u, "STRIDE_B must keep float4 alignment");

    __shared__ float sharedA[TILE_SIZE_N * STRIDE_A];
    __shared__ float sharedB[TILE_SIZE_N * STRIDE_B];

    // Identify the tile of C this thread block is responsible for
    const uint block_row = blockIdx.y;
    const uint block_column = blockIdx.x;

    constexpr uint WARP_STEPS_M = (WARP_TILE_M * WARP_TILE_K) / (WARPSIZE * ROWS_PER_THREAD * COLS_PER_THREAD * WARP_STEPS_K);
    // Warp subtile is WARP_SUB_M x WARP_SUB_K
    constexpr uint WARP_SUB_M = WARP_TILE_M / WARP_STEPS_M; // 64 / 1 = 64
    constexpr uint WARP_SUB_K = WARP_TILE_K / WARP_STEPS_K; // 64 / 4 = 16

    // Identify the warp tile position
    const uint warp_idx = threadIdx.x / WARPSIZE;
    const uint warps_per_row = TILE_SIZE_K / WARP_TILE_K;
    const uint warp_row = warp_idx / warps_per_row;
    const uint warp_col = warp_idx % warps_per_row;

    // Identify the thread position within the warp tile
    uint lane = threadIdx.x % WARPSIZE;
    const uint threads_per_subtile_row = WARP_SUB_K / COLS_PER_THREAD; // 16/4 = 4 threads per row
    const uint thread_row_in_sub = lane / threads_per_subtile_row;
    const uint thread_col_in_sub = lane % threads_per_subtile_row;

    const uint ty = thread_row_in_sub;
    const uint tx = thread_col_in_sub;

    // Move pointers from A[0], B[0] and C[0] to the starting positions of the tile
    A += block_row * TILE_SIZE_M * N;
    B += block_column * TILE_SIZE_K;
    C += (block_row * TILE_SIZE_M * K) + (block_column * TILE_SIZE_K);

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
    float reg_m[WARP_STEPS_M * ROWS_PER_THREAD];
    float reg_k[WARP_STEPS_K * COLS_PER_THREAD];

    const uint num_tiles = CEIL_DIV(N, TILE_SIZE_N);

    // Outer loop iterate over tiles
    for (int t = 0; t < num_tiles; t++) {
        // Populate smem using vectorised loads (We use offsets and is transposed)
        for (int load_offset = 0; load_offset < TILE_SIZE_M; load_offset += strideA) {
            const float4 v = reinterpret_cast<const float4*>(
                &A[(smem_ty_A + load_offset) * N + smem_tx_A * 4])[0];

            // Transpose A (instead of 128x16 previously for ex, now it will be 16x128)
            sharedA[(smem_tx_A * 4 + 0) * STRIDE_A + (smem_ty_A + load_offset)] = v.x;
            sharedA[(smem_tx_A * 4 + 1) * STRIDE_A + (smem_ty_A + load_offset)] = v.y;
            sharedA[(smem_tx_A * 4 + 2) * STRIDE_A + (smem_ty_A + load_offset)] = v.z;
            sharedA[(smem_tx_A * 4 + 3) * STRIDE_A + (smem_ty_A + load_offset)] = v.w;
        }

        // Load from as B as well but without transposing
        for (int load_offset = 0; load_offset < TILE_SIZE_N; load_offset += strideB) {
            reinterpret_cast<float4*>(
                &sharedB[(smem_ty_B + load_offset) * STRIDE_B + smem_tx_B * 4])[0] =
            reinterpret_cast<const float4*>(
                &B[(smem_ty_B + load_offset) * K + smem_tx_B * 4])[0];
        }

        __syncthreads();

        // Iterate over the shared dimension of the SMEM tiles
        for (int i = 0; i < TILE_SIZE_N; i++) {
            // Load slice at current i iteration in sharedA's register
            for (int wSubRow = 0; wSubRow < WARP_STEPS_M; wSubRow++) {
                uint base_row = (warp_row * WARP_TILE_M) + (wSubRow * WARP_SUB_M) + (ty * ROWS_PER_THREAD);

                // Each row "leader" load from sharedA, others will receive via shuffle
                const bool is_row_leader = (tx == 0);
                if (is_row_leader) {
                    #pragma unroll
                    for (int row = 0; row < ROWS_PER_THREAD; row += 4) {
                        const float4 va = reinterpret_cast<const float4*>(
                            &sharedA[i * STRIDE_A + base_row + row])[0];

                        reg_m[wSubRow * ROWS_PER_THREAD + row + 0] = va.x;
                        reg_m[wSubRow * ROWS_PER_THREAD + row + 1] = va.y;
                        reg_m[wSubRow * ROWS_PER_THREAD + row + 2] = va.z;
                        reg_m[wSubRow * ROWS_PER_THREAD + row + 3] = va.w;
                    }
                }

                for (int wSubCol = 0; wSubCol < WARP_STEPS_K; wSubCol++) {
                    uint col_base = (warp_col * WARP_TILE_K) + (wSubCol * WARP_SUB_K) + (tx * COLS_PER_THREAD);

                    // Only column leaders fill the register
                    const bool is_col_leader = (ty == 0);
                    if (is_col_leader) {
                        #pragma unroll
                        for (int col = 0; col < COLS_PER_THREAD; col += 4) {
                            const float4 vb = reinterpret_cast<const float4*>(
                                &sharedB[i * STRIDE_B + col_base + col])[0];

                            reg_k[wSubCol * COLS_PER_THREAD + col + 0] = vb.x;
                            reg_k[wSubCol * COLS_PER_THREAD + col + 1] = vb.y;
                            reg_k[wSubCol * COLS_PER_THREAD + col + 2] = vb.z;
                            reg_k[wSubCol * COLS_PER_THREAD + col + 3] = vb.w;
                        }
                    }
                    // broadcast from leader, then compute outer product
                    const unsigned mask = __activemask();
                    const uint row_leader_lane = ty * threads_per_subtile_row + 0;
                    const uint col_leader_lane = 0 * threads_per_subtile_row + tx;
                    #pragma unroll
                    for (int im = 0; im < ROWS_PER_THREAD; im++) {
                        float a_src = is_row_leader? reg_m[wSubRow * ROWS_PER_THREAD + im] : 0.0f;
                        float a_val = __shfl_sync(mask, a_src, row_leader_lane);
                        #pragma unroll
                        for (int ik = 0; ik < COLS_PER_THREAD; ik++) {
                            float b_src = is_col_leader ? reg_k[wSubCol * COLS_PER_THREAD + ik] : 0.f;
                            float b_val = __shfl_sync(mask, b_src, col_leader_lane);
                            const uint out_idx = (wSubRow * ROWS_PER_THREAD + im) * (WARP_STEPS_K * COLS_PER_THREAD) + (wSubCol * COLS_PER_THREAD + ik);
                            thread_results[out_idx] += a_val * b_val;
                        }
                    }
                }            
            }
        }
        __syncthreads();

        A += TILE_SIZE_N;  // Move right
        B += TILE_SIZE_N * K; // Move down
    }

    // Write results of the thread back to C
    float* C_tile_base = C; // Keep base pointer
    for (int wSubRow = 0; wSubRow < WARP_STEPS_M; wSubRow++) {
        for (int wSubCol = 0; wSubCol < WARP_STEPS_K; wSubCol++) {
            float* C_ptr = C_tile_base
                         + (warp_row * WARP_TILE_M + wSubRow * WARP_SUB_M) * K
                         + (warp_col * WARP_TILE_K + wSubCol * WARP_SUB_K);

            #pragma unroll
            for (int im = 0; im < ROWS_PER_THREAD; im++) {
                int c_row = ty * ROWS_PER_THREAD + im;
                #pragma unroll
                for (int ik = 0; ik < COLS_PER_THREAD; ik += 4) {
                    int c_col = tx * COLS_PER_THREAD + ik;

                    uint idx_out = (wSubRow * ROWS_PER_THREAD + im) * (WARP_STEPS_K * COLS_PER_THREAD)
                                   + (wSubCol * COLS_PER_THREAD + ik);

                    float4 temp_out;
                    temp_out.x = thread_results[idx_out + 0];
                    temp_out.y = thread_results[idx_out + 1];
                    temp_out.z = thread_results[idx_out + 2];
                    temp_out.w = thread_results[idx_out + 3];

                    float4 vc = reinterpret_cast<float4*>(&C_ptr[c_row * K + c_col])[0];

                    vc.x = alpha * temp_out.x + beta * vc.x;
                    vc.y = alpha * temp_out.y + beta * vc.y;
                    vc.z = alpha * temp_out.z + beta * vc.z;
                    vc.w = alpha * temp_out.w + beta * vc.w;

                    reinterpret_cast<float4*>(&C_ptr[c_row * K + c_col])[0] = vc;
                }
            }
        }
    }
}


void launch_sgemm_warptiling(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta) {
    constexpr int TILE_SIZE_M = 128;
    constexpr int TILE_SIZE_N = 16;
    constexpr int TILE_SIZE_K = 128;
    constexpr int WARP_TILE_M = 64;
    constexpr int WARP_TILE_K = 64;
    constexpr int ROWS_PER_THREAD = 4;
    constexpr int COLS_PER_THREAD = 4;
    constexpr int WARP_STEPS_K = 2;
    constexpr int NUM_THREADS = 128;

    dim3 blockDim(NUM_THREADS);
    dim3 gridDim(CEIL_DIV(K, TILE_SIZE_K), CEIL_DIV(M, TILE_SIZE_M));

    sgemm_warptiling<
        TILE_SIZE_M, TILE_SIZE_N, TILE_SIZE_K,
        WARP_TILE_M, WARP_TILE_K, WARP_STEPS_K,
        ROWS_PER_THREAD, COLS_PER_THREAD, NUM_THREADS>
        <<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * @brief Main entry point for the profiling program.
 */
int main(int argc, char** argv) {
    ResourceManager<float> resources;

    // Fixed size for profiling
    const int size = 8192;
    const float alpha = 0.5f;
    const float beta = 3.0f;
    
    const size_t mem_size = size * size * sizeof(float);

    std::cout << "Profiling sgemm_warptiling kernel with matrix size " << size << "x" << size << std::endl;

    // Allocate host memory
    float* A_host = (float*)malloc(mem_size);
    float* B_host = (float*)malloc(mem_size);
    float* C_host = (float*)malloc(mem_size);

    // Register host memory with resource manager
    resources.add_host_ptr(A_host);
    resources.add_host_ptr(B_host);
    resources.add_host_ptr(C_host);

    if (!A_host || !B_host || !C_host) {
        std::cerr << "Host memory allocation failed" << std::endl;
        return -1;
    }

    // Initialise matrices
    float* matrices[] = {A_host, B_host, C_host};
    initialiseArrays<float>(matrices, 3, size * size, -100.0f, 100.0f, 0);

    // Allocate device memory
    float* A_device;
    float* B_device;
    float* C_device;

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

    // Warm-up launches
    std::cout << "Running warm-up launches..." << std::endl;
    for (int i = 0; i < 2; ++i) {
        launch_sgemm_warptiling(A_device, B_device, C_device, size, size, size, alpha, beta);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Main kernel launch for profiling
    std::cout << "Running main kernel for profiling..." << std::endl;
    launch_sgemm_warptiling(A_device, B_device, C_device, size, size, size, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Kernel launch completed successfully" << std::endl;

    return 0;  // ResourceManager destructor will handle cleanups
}