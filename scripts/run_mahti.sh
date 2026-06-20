#!/bin/bash
#SBATCH --job-name=matmul_benchmark
#SBATCH --account=project_2019091.       #YOUR_PROJECT_HERE
#SBATCH --partition=gputest
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/matmul_%j.out
#SBATCH --error=logs/matmul_%j.err

# Load modules
module load gcc/11.2.0
module load cuda/11.5.0

# Create dirs
mkdir -p logs results

# Compile
echo "Compiling..."
nvcc -O3 -arch=sm_80 src/benchmark.cu -o benchmark -lcublas

# Run
echo "Running benchmark..."
./benchmark

# Plot (optional, needs matplotlib)
echo "Generating plots..."
module load python-data  # Mahti has a python-data module with matplotlib
python3 results/plot.py

echo "Done!"
