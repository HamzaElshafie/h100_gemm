# H100 GEMM (Project in progress)

A high-performance GEMM (General Matrix Multiply) implementation optimised for H100 GPUs.

**Blog:** https://hamzaelshafie.bearblog.dev/worklog-optimising-gemm-on-nvidia-h100-for-cublas-like-performance-wip/

| Kernel | TFLOP/s | Perf. relative to cuBLAS %<br>Full precision<br>(FP32) | Perf. relative to cuBLAS %<br>Mixed precision<br>(BF16 + FP32) |
|--------|---------|-------------------------------------------------------|---------------------------------------------------------------|
| Naive | 0.5/4.0 | 1 | 0.5 |
| Naive (coalesced) | 4.2/2.7 | 8.2 | 0.4 (Idk why!) |
| Tiled (SMEM) | 7.2/7.0 | 13.9 | 1.0 |
| 1D Register Tiling | 12.9/13.0 | 24.9 | 1.8 |
| 2D Register Tiling | 19.1/23.3 | 36.8 | 3.1 |
| Vectorised 2D Register Tiling | 37.2/25.6 | 72.0 | 3.3 |
| Warp Tiling | 41.4/31.5 | 79.8 | 4.3 |
| Tensor Cores (Async TMA + WGMMA) | NA/280.4 | NA | 37.8 |
| **cuBLAS** | **51.5 / 739.8** | **100%** | **100%** |

---

## Setup Instructions

### Option 1: Remote Setup (macOS)

If you are on macOS or do not have a local NVIDIA GPU:

1. **Rent a GPU instance** from a provider such as [Vast.ai](https://vast.ai).
2. **Start your instance**, add your **SSH public key**, and **copy** one of the SSH connection commands provided.
3. **Connect via SSH** using your preferred code editor (e.g., VS Code or Cursor):
   - In VS Code or Cursor, use the option to *Connect to Host...* and paste the SSH command.

Once connected to the remote instance, follow the setup steps below as you would on a local machine.

---

### Option 2: Local Setup (with NVIDIA GPU)

If you have an NVIDIA GPU available locally, begin with the steps below.

---

### 1. Fork & Clone the Repository

```bash
git clone https://github.com/HamzaElshafie/h100_gemm.git
cd h100_gemm
```

### 2. Install Prerequisites

#### Install Miniconda 
[Official installation instructions](https://www.anaconda.com/docs/getting-started/miniconda/install)

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

Close and repopen terminal, then run:

```bash
source ~/miniconda3/bin/activate
conda init --all
```

### 3. Create Environment from environment.yml
This will install all dependencies:

```bash
conda env create -f environment.yml
conda activate h100gemm_env
```

### (Optional) Configure NVCC for Your GPU

To ensure optimal performance and compatibility, configure NVCC with the correct compute capability for your GPU.

1. **Find your GPU's compute capability** [here](https://developer.nvidia.com/cuda-gpus)

2. **Update the CMake configuration**:  
   Open the `CMakeLists.txt` file in the root of the project and locate:

   ```cmake
   set(CMAKE_CUDA_ARCHITECTURES 90a)
   ```
   Replace 90a with the compute capability of your GPU (Hopper kernels will only work with 90a)

### 4. Build the Project

```bash
cmake -B build
cmake --build build
```

### 5. Run the Program
Navigate to the build directory and run the program:

```bash
cd build
./gemm general 0 bf16
```

replace `general` with your implementation (`general`/`hopper`/`cublas`) and `0` with your desired kernel ID.





