#pragma once

#include <iostream>
#include <launcher.cu>
#include <cublas_v2.h>

enum KernelID
{
    SIMON_naive_sgemm
};

void launchKernel(KernelID kernelID, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, 
    int M, int N, int K, float alpha, float beta, cublasHandle_t handle);

