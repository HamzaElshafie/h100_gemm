#!/bin/bash

# Script to calculate theoretical arithmetic intensity for GEMM operations

set -e

# Function to print usage
print_usage() {
    echo "Usage: $0 <implementation> <kernel_id>"
    echo "  Implementation: ampere | hopper | cublas"
    echo "  ID: 0, 1, 2, ..."
    echo "Example: $0 ampere 0"
}

# Function to calculate FLOPs for GEMM
calculate_flops() {
    local M=$1
    local N=$2
    local K=$3
    # FLOPs = 2*M*N*K + 3*M*N for alpha*(AB) + beta*C
    echo "scale=0; 2*$M*$N*$K + 3*$M*$N" | bc
}

# Function to calculate memory operations (in bytes)
calculate_memory() {
    local M=$1
    local N=$2
    local K=$3
    local bytes_per_element=4  # float = 4 bytes
    
    # Memory reads:
    # - Matrix A: M*N elements
    # - Matrix B: N*K elements
    # - Matrix C: M*K elements (for beta*C)
    # Memory writes:
    # - Matrix C: M*K elements (output)
    local total_elements=$(echo "scale=0; $M*$N + $N*$K + 2*$M*$K" | bc)
    echo "scale=0; $total_elements * $bytes_per_element" | bc
}

# Check arguments
if [ $# -ne 2 ]; then
    print_usage
    exit 1
fi

IMPL=$1
KERNEL_ID=$2

# Find the executable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_ROOT/build/gemm" ]; then
    EXECUTABLE="$PROJECT_ROOT/build/gemm"
elif [ -f "$PROJECT_ROOT/gemm" ]; then
    EXECUTABLE="$PROJECT_ROOT/gemm"
else
    echo "Error: Could not find gemm executable"
    echo "Make sure to build the project first"
    exit 1
fi

echo "Analyzing GEMM implementation: $IMPL, kernel ID: $KERNEL_ID"
echo "============================================================"

# Create temporary file for output
TEMP_DIR=$(mktemp -d)
REGULAR_OUTPUT="$TEMP_DIR/regular_output.txt"

# Cleanup function
cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Run benchmark
echo "Running benchmark..."
"$EXECUTABLE" "$IMPL" "$KERNEL_ID" > "$REGULAR_OUTPUT" 2>&1

if [ $? -ne 0 ]; then
    echo "Error: Benchmark failed"
    cat "$REGULAR_OUTPUT"
    exit 1
fi

# Parse results and calculate arithmetic intensity
echo ""
echo "Theoretical Arithmetic Intensity Analysis:"
echo "============================================================="

# Process each matrix size
while IFS= read -r line; do
    if [[ $line =~ Dimensions\ \(M\ =\ N\ =\ K\)\ =\ ([0-9]+) ]]; then
        size=${BASH_REMATCH[1]}
        M=$size
        N=$size
        K=$size
        
        # Calculate theoretical values
        flops=$(calculate_flops $M $N $K)
        memory_bytes=$(calculate_memory $M $N $K)
        ai=$(echo "scale=2; $flops / $memory_bytes" | bc)
        
        # Convert to more readable numbers
        flops_g=$(echo "scale=2; $flops / 1000000000" | bc)
        memory_gb=$(echo "scale=2; $memory_bytes / 1000000000" | bc)
        
        echo "Matrix Size ${M}x${N}x${K}:"
        echo "  Total FLOPs: $flops_g G"
        echo "  Total Memory: $memory_gb GB"
        echo "  Theoretical Arithmetic Intensity: $ai FLOPs/byte"
        
        # Extract actual performance from benchmark
        if [[ $(grep -A 1 "Dimensions (M = N = K) = $size" "$REGULAR_OUTPUT") =~ Average\ elapsed\ time:\ ([0-9.]+)\ s,\ TFLOPS:\ ([0-9.]+) ]]; then
            time=${BASH_REMATCH[1]}
            tflops=${BASH_REMATCH[2]}
            echo "  Achieved Performance: $tflops TFLOPS"
            echo "  Execution Time: $time seconds"
        fi
        echo ""
    fi
done < "$REGULAR_OUTPUT"

echo "Note: This is theoretical arithmetic intensity based on:"
echo "- Memory: Total bytes for reading A, B, C and writing C"
echo "- FLOPs: 2*M*N*K + 3*M*N (multiply-add for AB, alpha/beta multiplies, final add)" 