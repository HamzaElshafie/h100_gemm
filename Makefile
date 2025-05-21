CONDA_ENV_PATH ?= /root/miniconda3/envs/cuda_env
CUDA_PATH ?= $(CONDA_ENV_PATH)
HOST_COMPILER ?= /root/miniconda3/envs/cuda_env/bin/x86_64-conda-linux-gnu-g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)
BUILDDIR      := build

# RTX 4090 architecture (Ada Lovelace)
CUDA_ARCH := -arch=sm_89

# Common flags
CXXFLAGS   := -O3 -std=c++17
CUDA_FLAGS := $(CUDA_ARCH) \
             -O3 \
             -std=c++17 \
             --compiler-options "$(CXXFLAGS)"

# Include paths
INCLUDES := -I$(CUDA_PATH)/include -I./src -I./src/kernels -I./src/kernels/simon

# Source files
SOURCES   := $(shell find src -name '*.cu')
HEADERS   := $(shell find src -name '*.cuh' -o -name '*.h')
OBJECTS   := $(SOURCES:src/%.cu=$(BUILDDIR)/%.o)

# Libraries
LIBRARIES := -L$(CUDA_PATH)/lib64 -lcudart -lcublas

# Binary
TARGET    := sgemm

all: $(BUILDDIR)/$(TARGET)

$(BUILDDIR)/$(TARGET): $(OBJECTS)
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(CUDA_FLAGS) $(OBJECTS) -o $@ $(LIBRARIES)

$(BUILDDIR)/%.o: src/%.cu $(HEADERS)
	@mkdir -p $(dir $@)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(BUILDDIR)

.PHONY: all clean
