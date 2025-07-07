/**
 * @file utils.h
 * @brief Utility functions for CUDA development
 *
 * This header provides core utility functions focusing on error checking timing,
 * initialising arrays in memory and result comparison.
 */

#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>

/**
 * @brief CUDA error checking macro
 *
 * Evaluates a CUDA runtime call and checks for errors.
 * If an error is detected, prints detailed information and terminates the program.
 */
#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t error = call;                                                         \
        if (error != cudaSuccess) {                                                       \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "         \
                      << cudaGetErrorString(error) << " (" << error << ") " << std::endl; \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    } while (0)

/**
 * @brief cuBLAS error checking macro
 *
 * Evaluates a cuBLAS call and checks for errors.
 * If an error is detected, prints detailed information and terminates the program.
 */
#define CUBLAS_CHECK(call)                                                 \
    do                                                                     \
    {                                                                      \
        cublasStatus_t status = call;                                      \
        if (status != CUBLAS_STATUS_SUCCESS)                               \
        {                                                                  \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                      << " code " << status << std::endl;                  \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

/**
 * @brief Integer division with rounding up.
 *
 * Computes the ceiling of integer division between two numbers. Useful for calculating
 * the number of blocks needed to cover a size with a fixed block size.
 *
 * @param value The total number of items (e.g. threads, num_rows, num_columns).
 * @param divisor The size of each unit (e.g. block size).
 * @return The minimum number of units needed to fully cover the total value.
 */
#define CEIL_DIV(value, divisor) (((value) + (divisor) - 1) / (divisor))

/**
 * @brief Initialise multiple arrays with random values in a specified range
 *
 * @param arrays     Array of pointers to initialize
 * @param num_arrays Number of arrays to initialize
 * @param size       Number of elements in each array
 * @param min       Minimum value for random numbers (default: 0.0)
 * @param max       Maximum value for random numbers (default: 1.0)
 * @param seed       Seed for random generator, 0 means use time(0) (default: 0)
 */
inline void initialiseArrays(float** arrays, int num_arrays, size_t size, float min = -1.0f, float max = 1.0f, unsigned int seed = 0) {
    // Set random seed
    if (seed == 0) {
        seed = static_cast<unsigned int>(time(0)); // get current time
    }
    srand(seed);

    float range = max - min;

    for (int i = 0; i < num_arrays; i++) { // Iterate through each array pointer
        for (size_t j = 0; j < size; j++) { // Iterate through each element
            arrays[i][j] = min + (static_cast<float>(rand()) / RAND_MAX) * range;
        }
    }
}

/**
 * @brief Measure CPU execution time using std::chrono
 *
 * @tparam Func     Function type
 * @param function  Function or lambda to measure
 * @return double   Execution time in milliseconds
 */
template <typename Function>
double measureExecutionTime(Function function) {
    auto start = std::chrono::steady_clock::now();
    function();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = (end - start);
    return duration.count();
}

/**
 * @brief Measure GPU kernel execution time using CUDA events.
 *
 * @note This function is ideal for timing a standalone kernel launch. For benchmarking repeated kernel executions, consider running the kernel multiple times in a loop and averaging the timings for more accurate results.
 *
 * @tparam KernelFunc  Kernel function type
 * @param kernel       Kernel function to measure
 * @return float       Execution time in milliseconds
 */
template <typename KernelFunc>
float measureKernelTime(KernelFunc kernel) {
    cudaEvent_t start;
    cudaEvent_t stop;
    float elapsed_time;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Start stopwatch
    CUDA_CHECK(cudaEventRecord(start));
    // Launch kernel
    kernel();
    // Stop stopwatch
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

    // Free events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return elapsed_time;
}

/**
 * @brief Compare two arrays of floating-point results using absolute and relative tolerance.
 *
 * Compares a reference array to a test array, element-wise, using both absolute and relative tolerances.
 * Prints the first mismatch found.
 *
 * @param ref_output  Reference result array.
 * @param test_output Test result array.
 * @param size        Number of elements to compare.
 * @param atol        Absolute tolerance (default: 1e-4).
 * @param rtol        Relative tolerance (default: 1e-5).
 * @return bool       True if all elements match within tolerances, false otherwise.
 */
inline bool compareResults(const float *ref_output, const float *test_output, size_t size, float atol = 1e-1f, float rtol = 1e-1f) {
    for (size_t i = 0; i < size; i++) {
        float a = ref_output[i];
        float b = test_output[i];
        float abs_diff = std::fabs(a - b);
        float rel_diff = abs_diff / (std::fabs(a) + 1e-6f);

        if (abs_diff > atol && rel_diff > rtol) {
            std::cout << "Mismatch at index " << i
                      << ": Reference = " << a
                      << ", Test = " << b
                      << ", abs diff = " << abs_diff
                      << ", rel diff = " << rel_diff
                      << std::endl;
            return false;
        }
    }
    return true;
}

/**
 * @brief Resource manager to track and cleanup CUDA/CPU resources
 *
 * Provides RAII-style management of CUDA and CPU resources including:
 * - Host memory pointers
 * - Device memory pointers
 * - CUDA events
 * - cuBLAS handles
 *
 * Resources are automatically freed when the manager goes out of scope.
 */
class ResourceManager {

private:
    std::vector<float*> host_ptrs;
    std::vector<float*> device_ptrs;
    std::vector<cudaEvent_t> events;
    cublasHandle_t* cublas_handle;

public:
    ResourceManager() : cublas_handle(nullptr) {}

    /**
     * @brief Add a host memory pointer to be managed
     * @param ptr Pointer to host memory
     */
    void add_host_ptr(float* ptr) {
        if (ptr) {
            host_ptrs.push_back(ptr);
        }
    }

    /**
     * @brief Add a device memory pointer to be managed
     * @param ptr Pointer to device memory
     */
    void add_device_ptr(float* ptr) {
        if (ptr) {
            device_ptrs.push_back(ptr);
        }
    }

    /**
     * @brief Add a CUDA event to be managed
     * @param event CUDA event handle
     */
    void add_event(cudaEvent_t event) {
        events.push_back(event);
    }

    /**
     * @brief Set the cuBLAS handle to be managed
     * @param handle Pointer to cuBLAS handle
     */
    void set_cublas_handle(cublasHandle_t* handle) {
        cublas_handle = handle;
    }

    /**
     * @brief Destructor that frees all managed resources
     */
    ~ResourceManager() {

        // Free host memory
        for (auto ptr : host_ptrs) {
            if (ptr) {
                free(ptr);
            }
        }

        // Free device memory
        for (auto ptr : device_ptrs) {
            if (ptr) {
                CUDA_CHECK(cudaFree(ptr));
            }
        }

        // Destroy events
        for (auto event : events) {
            CUDA_CHECK(cudaEventDestroy(event));
        }

        // Destroy cuBLAS handle
        if (cublas_handle) {
            cublasDestroy(*cublas_handle);
        }
    }
};
#endif
