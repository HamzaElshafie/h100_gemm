#!/bin/bash

# Wrapper script to run GEMM benchmarks with Nsight Compute profiling
# to calculate real arithmetic intensity based on actual memory transfers.

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

echo "Profiling GEMM implementation: $IMPL, kernel ID: $KERNEL_ID"
echo "============================================================"

# Create temporary files
TEMP_DIR=$(mktemp -d)
REGULAR_OUTPUT="$TEMP_DIR/regular_output.txt"
NSIGHT_CSV="$TEMP_DIR/nsight_profile.csv"

# Cleanup function
cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Run regular benchmark first
echo "Running regular benchmark..."
"$EXECUTABLE" "$IMPL" "$KERNEL_ID" > "$REGULAR_OUTPUT" 2>&1

if [ $? -ne 0 ]; then
    echo "Error: Regular benchmark failed"
    cat "$REGULAR_OUTPUT"
    exit 1
fi

# Run with Nsight Compute profiling
echo "Running with Nsight Compute profiling..."

# Nsight Compute metrics for memory analysis
METRICS="dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum"

nv-nsight-cu-cli \
    --metrics "$METRICS" \
    --target-processes application-only \
    --export "$NSIGHT_CSV" \
    --force-overwrite \
    "$EXECUTABLE" "$IMPL" "$KERNEL_ID" > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "Error: Nsight Compute profiling failed"
    echo "Make sure nv-nsight-cu-cli is installed and available in PATH"
    exit 1
fi

# Parse results and combine them
echo ""
echo "Combined Results with Real Arithmetic Intensity:"
echo "================================================================================"

# Matrix sizes used in the benchmark
SIZES=(512 1024 2048 4096 8192)

# Extract benchmark results and combine with profiling data
size_idx=0
while IFS= read -r line; do
    if [[ $line =~ Dimensions\ \(M\ =\ N\ =\ K\)\ =\ ([0-9]+) ]]; then
        current_size=${BASH_REMATCH[1]}
        
        # Find the corresponding size index
        for i in "${!SIZES[@]}"; do
            if [[ "${SIZES[$i]}" -eq "$current_size" ]]; then
                size_idx=$((i + 1))  # 1-based for CSV parsing
                break
            fi
        done
        
    elif [[ $line =~ Average\ elapsed\ time:\ ([0-9.]+)\ s,\ TFLOPS:\ ([0-9.]+),\ cuBLAS\ TFLOPS:\ ([0-9.]+),\ Performance\ relative\ to\ cuBLAS:\ ([0-9.]+)% ]]; then
        avg_time=${BASH_REMATCH[1]}
        tflops=${BASH_REMATCH[2]}
        cublas_tflops=${BASH_REMATCH[3]}
        perf_ratio=${BASH_REMATCH[4]}
        
        # Extract memory data from CSV for this kernel run
        # Skip header and get the row for this size
        if [ -f "$NSIGHT_CSV" ] && [ $size_idx -gt 0 ]; then
            # Get the data row (header + size_idx)
            csv_row=$(sed -n "$((size_idx + 1))p" "$NSIGHT_CSV")
            
            if [ -n "$csv_row" ]; then
                # Parse CSV row - assumes format: "ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time","Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"
                # We need to extract the bytes read and write values
                
                # Extract bytes read (assuming it's in the CSV)
                bytes_read=$(echo "$csv_row" | grep -o '"dram__bytes_read.sum","[^"]*","[^"]*"' | cut -d'"' -f6 || echo "0")
                bytes_write=$(echo "$csv_row" | grep -o '"dram__bytes_write.sum","[^"]*","[^"]*"' | cut -d'"' -f6 || echo "0")
                
                # Alternative parsing if the above doesn't work - try awk
                if [ "$bytes_read" == "0" ] && [ "$bytes_write" == "0" ]; then
                    # This is a simplified approach - may need adjustment based on actual CSV format
                    bytes_read=$(awk -F',' -v row=$size_idx 'NR==row+1 && /dram__bytes_read/ {gsub(/"/, "", $NF); print $NF}' "$NSIGHT_CSV" | head -1)
                    bytes_write=$(awk -F',' -v row=$size_idx 'NR==row+1 && /dram__bytes_write/ {gsub(/"/, "", $NF); print $NF}' "$NSIGHT_CSV" | head -1)
                fi
                
                # Default to 0 if parsing failed
                bytes_read=${bytes_read:-0}
                bytes_write=${bytes_write:-0}
                
                # Calculate total bytes and arithmetic intensity
                total_bytes=$(echo "scale=0; $bytes_read + $bytes_write" | bc)
                
                if [ "$total_bytes" != "0" ]; then
                    flops=$(calculate_flops $current_size $current_size $current_size)
                    ai=$(echo "scale=1; $flops / $total_bytes" | bc)
                else
                    ai="N/A"
                fi
                
                # Convert bytes to GB for display
                bytes_read_gb=$(echo "scale=2; $bytes_read / 1000000000" | bc)
                bytes_write_gb=$(echo "scale=2; $bytes_write / 1000000000" | bc)
                total_bytes_gb=$(echo "scale=2; $total_bytes / 1000000000" | bc)
                
                echo "Size ${current_size}x${current_size}x${current_size}:"
                echo "  Time: $avg_time s"
                echo "  Custom TFLOPS: $tflops"
                echo "  cuBLAS TFLOPS: $cublas_tflops"
                echo "  Performance vs cuBLAS: $perf_ratio%"
                echo "  Memory Read: $bytes_read_gb GB"
                echo "  Memory Write: $bytes_write_gb GB"
                echo "  Total Memory: $total_bytes_gb GB"
                echo "  Real Arithmetic Intensity: $ai FLOPs/byte"
                echo ""
            else
                echo "Size ${current_size}x${current_size}x${current_size}: No profiling data available"
                echo ""
            fi
        fi
    fi
done < "$REGULAR_OUTPUT"

echo "Note: If arithmetic intensity shows as N/A, the CSV parsing may need adjustment"
echo "You can examine the raw CSV file format and update the parsing logic accordingly" 