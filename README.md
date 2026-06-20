# CUDA Matrix Multiplication: Naive vs Tiled vs cuBLAS

## Overview

This project benchmarks three implementations of matrix multiplication on GPU:

1. **Naive CUDA kernel** — where each thread computes one output element, reading it directly from the global memory.
2. **Tiled CUDA kernel** — that uses shared memory tiling (32×32 tiles) in order to reduce global memory accesses and improve the cache reuse.
3. **cuBLAS** — that is NVIDIA's highly optimised BLAS library, used as the SOTA baseline.

The key insight is that the bottleneck in the naive approach is **global memory bandwidth**. So then by loading tiles into shared memory (which is ~100× faster than global memory), the tiled kernel reduces in a significant way the memory traffic and it also achieves higher throughput.

## Project Structure

```
.
├── src/
│   ├── benchmark.cu       # All kernels + main benchmark loop
│   ├── matmul_naive.cu    # Standalone naive kernel
│   ├── matmul_tiled.cu    # Standalone tiled kernel
│   └── matmul_cublas.cu   # Standalone cuBLAS wrapper
├── results/
│   ├── plot.py            # Generate plots from CSV
│   └── benchmark.csv      # Output data (generated at runtime)
├── scripts/
│   └── run_mahti.sh       # SLURM job script for Mahti
├── Makefile
└── README.md
```

## How to Run on Mahti (CSC)

### 1. Connect and clone
```bash
ssh <user>@mahti.csc.fi
git clone <your-repo-url>
cd Parallel_Programming_finalproject
```

### 2. Edit the SLURM script
```bash
nano scripts/run_mahti.sh
# Change YOUR_PROJECT_HERE to your CSC project (e.g. project_2001234)
```

### 3. Submit the job
```bash
mkdir -p logs
sbatch scripts/run_mahti.sh
```

### 4. Monitor
```bash
squeue -u $USER
# When finished:
cat logs/matmul_<jobid>.out
```

### 5. Plot results
```bash
module load python-data
python3 results/plot.py
```

## How to Run Locally (if CUDA available)
```bash
make run    # compiles and runs
make plot   # also generates plots
```

## Expected Output

```
GPU: NVIDIA A100-SXM4-40GB
Compute Capability: 8.0
Global Memory: 40.0 GB

N= 256 | Naive:    0.18 ms ( 188.0 GFLOPS) | Tiled:    0.05 ms ( 680.0 GFLOPS) | cuBLAS:   0.03 ms (1120.0 GFLOPS) | Speedup(T/N): 3.62x
N= 512 | Naive:    1.20 ms ( 224.0 GFLOPS) | Tiled:    0.31 ms ( 869.0 GFLOPS) | cuBLAS:   0.11 ms (2430.0 GFLOPS) | Speedup(T/N): 3.87x
N=1024 | Naive:    9.10 ms ( 235.0 GFLOPS) | Tiled:    2.10 ms (1018.0 GFLOPS) | cuBLAS:   0.42 ms (5100.0 GFLOPS) | Speedup(T/N): 4.33x
N=2048 | Naive:   71.00 ms ( 242.0 GFLOPS) | Tiled:   14.50 ms (1182.0 GFLOPS) | cuBLAS:   1.30 ms (13200.0 GFLOPS) | Speedup(T/N): 4.90x
N=4096 | Naive:  560.00 ms ( 245.0 GFLOPS) | Tiled:  110.00 ms (1250.0 GFLOPS) | cuBLAS:   5.60 ms (24500.0 GFLOPS) | Speedup(T/N): 5.09x
```

## Key Concepts

### Why tiling works
- **Global memory** latency: ~400–800 cycles
- **Shared memory** latency: ~20–30 cycles
- Each element in a tile is loaded once from global memory and reused `TILE_SIZE` times
- This reduces global memory bandwidth by a factor of `TILE_SIZE` (32×)

### Arithmetic Intensity
- Naive: reads `2N` floats per output element → low arithmetic intensity
- Tiled: reuses each float `TILE_SIZE` times → much higher arithmetic intensity

## Requirements
- CUDA 11.x or later
- cuBLAS (included with CUDA toolkit)
- Python 3 + pandas + matplotlib (for plots)
