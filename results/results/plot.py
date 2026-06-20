import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

os.makedirs("results", exist_ok=True)

df = pd.read_csv("results/benchmark.csv")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("CUDA Matrix Multiplication Benchmark (NVIDIA A100)", fontsize=14, fontweight='bold')

colors = {'Naive': '#e74c3c', 'Tiled': '#3498db', 'cuBLAS': '#2ecc71'}

# ── Plot 1: Execution time ────────────────────────────────────────────────────
ax = axes[0]
ax.plot(df['N'], df['naive_ms'],  'o-', color=colors['Naive'],  label='Naive',  linewidth=2, markersize=6)
ax.plot(df['N'], df['tiled_ms'],  's-', color=colors['Tiled'],  label='Tiled',  linewidth=2, markersize=6)
ax.plot(df['N'], df['cublas_ms'], '^-', color=colors['cuBLAS'], label='cuBLAS', linewidth=2, markersize=6)
ax.set_xlabel('Matrix Size (N×N)', fontsize=11)
ax.set_ylabel('Time (ms)', fontsize=11)
ax.set_title('Execution Time', fontsize=12)
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(df['N'])
ax.set_xticklabels([f'{n}' for n in df['N']], rotation=45)

# ── Plot 2: GFLOPS ────────────────────────────────────────────────────────────
ax = axes[1]
ax.plot(df['N'], df['naive_gflops'],  'o-', color=colors['Naive'],  label='Naive',  linewidth=2, markersize=6)
ax.plot(df['N'], df['tiled_gflops'],  's-', color=colors['Tiled'],  label='Tiled',  linewidth=2, markersize=6)
ax.plot(df['N'], df['cublas_gflops'], '^-', color=colors['cuBLAS'], label='cuBLAS', linewidth=2, markersize=6)
ax.set_xlabel('Matrix Size (N×N)', fontsize=11)
ax.set_ylabel('GFLOPS', fontsize=11)
ax.set_title('Throughput (GFLOPS)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(df['N'])
ax.set_xticklabels([f'{n}' for n in df['N']], rotation=45)

# ── Plot 3: Speedup over naive ────────────────────────────────────────────────
ax = axes[2]
speedup_tiled  = df['naive_ms'] / df['tiled_ms']
speedup_cublas = df['naive_ms'] / df['cublas_ms']

x = np.arange(len(df['N']))
width = 0.35
bars1 = ax.bar(x - width/2, speedup_tiled,  width, label='Tiled vs Naive',  color=colors['Tiled'],  alpha=0.85)
bars2 = ax.bar(x + width/2, speedup_cublas, width, label='cuBLAS vs Naive', color=colors['cuBLAS'], alpha=0.85)

# Add value labels on bars
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{bar.get_height():.1f}x', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{bar.get_height():.1f}x', ha='center', va='bottom', fontsize=8)

ax.axhline(y=1, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Matrix Size (N×N)', fontsize=11)
ax.set_ylabel('Speedup (×)', fontsize=11)
ax.set_title('Speedup over Naive Kernel', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([f'{n}' for n in df['N']], rotation=45)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("results/benchmark_plots.png", dpi=150, bbox_inches='tight')
print("Saved: results/benchmark_plots.png")
plt.show()

# ── Summary table ──────────────────────────────────────────────────────────────
print("\n=== Summary Table ===")
print(f"{'N':>6} | {'Naive (ms)':>10} | {'Tiled (ms)':>10} | {'cuBLAS (ms)':>11} | {'Speedup T/N':>11} | {'Speedup C/N':>11}")
print("-" * 72)
for _, row in df.iterrows():
    st = row['naive_ms'] / row['tiled_ms']
    sc = row['naive_ms'] / row['cublas_ms']
    print(f"{int(row['N']):>6} | {row['naive_ms']:>10.2f} | {row['tiled_ms']:>10.2f} | "
          f"{row['cublas_ms']:>11.2f} | {st:>11.2f}x | {sc:>11.2f}x")
