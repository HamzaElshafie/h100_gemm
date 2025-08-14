/**
 * @file runner.cu
 * @brief Kernel launcher implementation.
 *
 * Dispatches the requested kernel to the launchers.
 */
#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>

#include "runner.cuh"
#include "kernels/ampere/launcher.cuh"
#include "kernels/hopper/launcher.cuh"
