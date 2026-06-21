#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 16
#define TILE_SIZE 32

// ─── Naive kernel ────────────────────────────────────────────────────────────
__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// ─── Tiled kernel ────────────────────────────────────────────────────────────
__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        int A_col = t * TILE_SIZE + threadIdx.x;
        int B_row = t * TILE_SIZE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < N && A_col < N) ? A[row * N + A_col] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (B_row < N && col < N) ? B[B_row * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE_SIZE; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N)
        C[row * N + col] = sum;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────
void init_matrix(float* M, int N) {
    for (int i = 0; i < N * N; i++)
        M[i] = (float)rand() / RAND_MAX;
}

// Returns max absolute difference between two matrices (correctness check)
float max_diff(float* A, float* B, int N) {
    float max_d = 0.0f;
    for (int i = 0; i < N * N; i++) {
        float d = fabsf(A[i] - B[i]);
        if (d > max_d) max_d = d;
    }
    return max_d;
}

// Compute GFLOPS: 2*N^3 floating point ops
double gflops(int N, float ms) {
    return (2.0 * N * N * N) / (ms * 1e6);
}

// ─── Benchmark runner ─────────────────────────────────────────────────────────
void benchmark(int N) {
    size_t size = N * N * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C_naive  = (float*)malloc(size);
    float *h_C_tiled  = (float*)malloc(size);
    float *h_C_cublas = (float*)malloc(size);

    init_matrix(h_A, N);
    init_matrix(h_B, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;

    // ── Warm-up (avoid cold-start bias) ──
    dim3 block_n(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_n((N+BLOCK_SIZE-1)/BLOCK_SIZE, (N+BLOCK_SIZE-1)/BLOCK_SIZE);
    matmul_naive<<<grid_n, block_n>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // ── Naive ──
    cudaEventRecord(start);
    matmul_naive<<<grid_n, block_n>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float ms_naive = ms;
    cudaMemcpy(h_C_naive, d_C, size, cudaMemcpyDeviceToHost);

    // ── Tiled ──
    dim3 block_t(TILE_SIZE, TILE_SIZE);
    dim3 grid_t((N+TILE_SIZE-1)/TILE_SIZE, (N+TILE_SIZE-1)/TILE_SIZE);
    cudaEventRecord(start);
    matmul_tiled<<<grid_t, block_t>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float ms_tiled = ms;
    cudaMemcpy(h_C_tiled, d_C, size, cudaMemcpyDeviceToHost);

   
    // ── cuBLAS ──
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f, beta = 0.0f;

    // Warm-up cuBLAS to avoid measuring initialization overhead
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
            &alpha, d_B, N, d_A, N, &beta, d_C, N);
    cudaDeviceSynchronize();

    // Timed cuBLAS run
    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
            &alpha, d_B, N, d_A, N, &beta, d_C, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float ms_cublas = ms;
    cudaMemcpy(h_C_cublas, d_C, size, cudaMemcpyDeviceToHost);
    cublasDestroy(handle);

    // ── Results ──
    float diff_tiled  = max_diff(h_C_naive, h_C_tiled,  N);
    float diff_cublas = max_diff(h_C_naive, h_C_cublas, N);

    printf("N=%4d | Naive: %8.2f ms (%6.1f GFLOPS) | "
           "Tiled: %8.2f ms (%6.1f GFLOPS) | "
           "cuBLAS: %7.2f ms (%6.1f GFLOPS) | "
           "Speedup(T/N): %.2fx | "
           "Diff(tiled): %.2e | Diff(cublas): %.2e\n",
           N,
           ms_naive,  gflops(N, ms_naive),
           ms_tiled,  gflops(N, ms_tiled),
           ms_cublas, gflops(N, ms_cublas),
           ms_naive / ms_tiled,
           diff_tiled, diff_cublas);

    // Save CSV row for plotting
    FILE* f = fopen("results/benchmark.csv", "a");
    if (f) {
        fprintf(f, "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                N,
                ms_naive,  gflops(N, ms_naive),
                ms_tiled,  gflops(N, ms_tiled),
                ms_cublas, gflops(N, ms_cublas));
        fclose(f);
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(h_A); free(h_B);
    free(h_C_naive); free(h_C_tiled); free(h_C_cublas);
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main() {
    printf("CUDA Matrix Multiplication Benchmark\n");
    printf("=====================================\n\n");

    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Global Memory: %.1f GB\n\n", prop.totalGlobalMem / 1e9);

    // Write CSV header
    FILE* f = fopen("results/benchmark.csv", "w");
    if (f) {
        fprintf(f, "N,naive_ms,naive_gflops,tiled_ms,tiled_gflops,cublas_ms,cublas_gflops\n");
        fclose(f);
    }

    int sizes[] = {256, 512, 1024, 2048, 4096};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        benchmark(sizes[i]);
    }

    printf("\nDone. Results saved to results/benchmark.csv\n");
    return 0;
}
