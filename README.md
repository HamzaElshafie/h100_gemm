# H100 GEMM (Project in progress)

A high-performance GEMM (General Matrix Multiply) implementation optimised for H100 GPUs.

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

#### CUDA Tools
For profiling capabilities (arithmetic intensity, memory analysis), install Nsight Compute:
```bash
apt install nsight-compute-2025.2.1
```

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
   set(CMAKE_CUDA_ARCHITECTURES 90)
   ```
   Replace 90 with the compute capability of your GPU

### 4. Build the Project

```bash
cmake -B build
cmake --build build
```

### 5. Run the Program
Navigate to the build directory and run the program:

```bash
cd build
./gemm ampere/hopper 0
```

Replace ampere and 0 with desired arguments.

### 6. Profiling with Nsight Compute
To analyse performance metrics like arithmetic intensity and memory transfers:

```bash
# Make sure you're in the project root directory
cd /path/to/h100_gemm

# Run the profiling script
./scripts/profile_with_ai.sh ampere 0
```

Again replace `ampere` with your implementation (`ampere`/`hopper`/`cublas`) and `0` with your desired kernel ID.

The script will automatically test different matrix sizes (512, 1024, 2048, 4096, 8192) and provide comprehensive performance analysis for each.
