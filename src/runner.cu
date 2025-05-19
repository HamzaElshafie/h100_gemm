#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>

#include "launcher.cuh"

void launchKernel(KernelID kernelID, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, 
    int M, int N, int K, float alpha, float beta, cublasHandle_t handle)
    {
        switch(kernelID)
        {
            case SIMON_naive_sgemm:
                simon::run_sgemm_naive(A, B, C, M, N, K, alpha, beta);
                break;
            default:
                throw std::invalid_argument("Unknown kernel ID");
        }
    }