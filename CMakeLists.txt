cmake_minimum_required(VERSION 3.18)
project(h100_gemm CUDA CXX)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Find required packages
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CUDA_INCLUDE_DIRS})

# Collect source files
file(GLOB_RECURSE SOURCES 
    "src/*.cu"
    "src/*.cuh"
    "src/*.h"
)

# Create executable
add_executable(gemm ${SOURCES})

# Link libraries
target_link_libraries(gemm PRIVATE 
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
)

# Set C++ standard
set_target_properties(gemm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
