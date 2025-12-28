cmake_minimum_required(VERSION 3.18)
project(h100_gemm CUDA CXX)

# Set CUDA architecture to 90a (Hopper with Acceleration)
set(CMAKE_CUDA_ARCHITECTURES 90a)

# Ensure the correct compiler is used from your environment
set(CMAKE_CUDA_COMPILER /root/miniconda3/envs/h100gemm_env/bin/nvcc)

# Optimization level and Host compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -fPIC")

# --ptxas-options=-v is crucial here. 
# It will print register usage and spill info to your terminal during build.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Collect source files
file(GLOB_RECURSE SOURCES 
    "src/*.cu"
    "src/*.cuh"
    "src/*.h"
)

# Create executable
add_executable(gemm ${SOURCES})

# Link libraries
# Note: We use the modern CUDAToolkit targets
target_link_libraries(gemm PRIVATE 
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    CUDA::cuda_driver
)

# Set C++ standard and Disable Separable Compilation
# Disabling separable compilation allows the compiler to perform 
# cross-function optimizations and hoisting of WGMMA descriptors.
set_target_properties(gemm PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)