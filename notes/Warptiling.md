# Kernel 7: Warp Tiling

So far we exploited two *levels* of paralellism. 

1. **Block tiling**: Each thread block computed a large tile of the output matrix C, reusing tiles of A and B from shared memory.
2. **Register tiling**: Each thread computed a small sub-tile of C (ROWS_PER_THREAD × COLS_PER_THREAD) entirely in registers, maximising data reuse before writing results back to
   global memory.

For this kernel, we will introduce a new level of tiling between block tiling and thread tiling and that is warp tiling.

## What are warps? 

In CUDA, a warp is a group of 32 threads that are scheduled together and execute in parallel.
All threads in a warp run on the same Streaming Multiprocessor (SM), and each SM executes many warps at the same time.

Warps are the fundamental (Atomic level) execution unit on NVIDIA GPUs:
-	All threads in a warp execute the same instruction at the same time (SIMT: Single Instruction, Multiple Threads).
- Each thread in the warp operates on its own registers and memory addresses.

When a warp issues an instruction (memory or even arithemtic), the result might take multiple cycles to become available. Instead of waiting, the **warp scheduler** switches to another warp that is ready to run. This latency hiding is how GPUs keep their execution units busy. Hopper 
SMs have 4 warp schedulers per SM.

## Warptiling

Warp tiling sits between block tiling and thread tiling in the optimisation hierarchy. Instead of having all threads in a block cooperatively work on one large tile, we partition that 
tile into smaller sub-tiles, each assigned to a warp. This turns the warp into the middle-level unit of computation.

This extra level of tiling provides several benefits:

### Alignment with hardware scheduling:
   
The warp is the fundamental execution unit in NVIDIA GPUs. By giving each warp its own sub-tile of the output, we align our work partitioning
with the way the hardware actually schedules instructions.

  - Each warp can execute independently.
  - If one warp stalls on memory, others can continue executing, which keeps warp scheduler slots full and reduces idle cycles.
  
### Control over shared memory access:

Shared memory in each SM is divided into 32 banks, each 4 bytes wide. Every time a warp accesses shared memory, each thread gets mapped to one of these banks, based on the address (or word index) it wants.

- Bank 0 holds all words whose indices are divisible by 32 (word 0, 32, 64, …).
- Bank 1 holds words with index 1, 33, 65, … and so on.
- The bank is determined by:
   - bank_index = word_index % 32

### Improved register cache locality:
   
The register file (RF) inside each Streaming Multiprocessor stores per-thread variables. On Hopper, it’s split into multiple single‑ported banks.
A bank can only serve one access per cycle. If two threads in the same warp try to read from the same bank in the same cycle, the accesses are 
serialized. This is called a bank conflict and it increases the time it takes to fetch operands for an instruction.

Between the RF and the execution units are Operand Collector Units (OCUs) [(Esfeden et al., 2020b)](https://microarch.org/micro53/papers/738300a996.pdf).
Each OCU fetches source operands from the register banks and stores them in a small buffer, with space for three 128‑byte entries. If an operand is 
needed again soon, it can be served directly from this buffer instead of going back to the main RF. This avoids both bank conflicts and extra RF traffic.

Warp tiling helps here because each warp works on a small, fixed sub‑tile of the output matrix, so it tends to reuse the same registers repeatedly 
in the inner loop. This makes bank conflicts less likely and increases the chances that operands can be reused directly from the OCU buffer.

