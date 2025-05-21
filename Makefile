CUDA_PATH ?= /usr/local/cuda
HOST_COMPILER ?= g++
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
INCLUDES  := -I$(CUDA_PATH)/include -I./src

# Source files
SOURCES   := $(wildcard src/*.cu)
HEADERS   := $(wildcard src/*.cuh src/*.h)
OBJECTS   := $(SOURCES:src/%.cu=$(BUILDDIR)/%.o)

# Libraries
LIBRARIES := -lcudart -lcublas

# Binary
TARGET    := sgemm

all: $(BUILDDIR)/$(TARGET)

$(BUILDDIR)/$(TARGET): $(OBJECTS)
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(CUDA_FLAGS) $(OBJECTS) -o $@ $(LIBRARIES)

$(BUILDDIR)/%.o: src/%.cu $(HEADERS)
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(BUILDDIR)

.PHONY: all clean
