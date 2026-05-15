// Matrix Multiplication using CUDA (Global and Shared Memory)
// Compile: nvcc 4_matrix_mul.cu -o 4_matrix_mul
// Run:     ./4_matrix_mul

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

#define TILE 16

#define CHECK(c) { \
    cudaError_t e = (c); \
    if (e != cudaSuccess) { \
        cout << "CUDA error: " << cudaGetErrorString(e) << endl; \
        exit(1); \
    } \
}

__global__ void matMulGlobal(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0;
        for (int k = 0; k < n; k++)
            sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

__global__ void matMulShared(float *A, float *B, float *C, int n) {
    __shared__ float tA[TILE][TILE];
    __shared__ float tB[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0;
    for (int t = 0; t < (n + TILE - 1) / TILE; t++) {
        if (row < n && t * TILE + threadIdx.x < n)
            tA[threadIdx.y][threadIdx.x] = A[row * n + t * TILE + threadIdx.x];
        else
            tA[threadIdx.y][threadIdx.x] = 0;
        if (col < n && t * TILE + threadIdx.y < n)
            tB[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * n + col];
        else
            tB[threadIdx.y][threadIdx.x] = 0;
        __syncthreads();
        for (int k = 0; k < TILE; k++)
            sum += tA[threadIdx.y][k] * tB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < n && col < n)
        C[row * n + col] = sum;
}

void cpuMatMul(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int k = 0; k < n; k++)
                sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
    }
}

int main() {
    cout << "Neeti Kurulkar BE A 41038" << endl << endl;

    int smCount;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);

    float *tmp;
    cudaMalloc(&tmp, sizeof(float));
    cudaDeviceSynchronize();
    cudaFree(tmp);

    int mat_sizes[] = {64, 128, 256, 512};
    int num_sizes = 4;

    ofstream csv("4_matrix_mul.csv");
    csv << "size,seq_time,par_global,par_shared,speedup_global,speedup_shared,efficiency\n";

    cout << "N\tSeq(s)\t\tGlobal(s)\tShared(s)\tSp(Gl)\t\tSp(Sh)\t\tEfficiency" << endl;

    for (int t = 0; t < num_sizes; t++) {
        int n = mat_sizes[t];
        int tot = n * n;

        float *h_A = new float[tot];
        float *h_B = new float[tot];
        float *h_C = new float[tot]();
        for (int i = 0; i < tot; i++) {
            h_A[i] = rand() % 10;
            h_B[i] = rand() % 10;
        }

        double cpu_start = (double)clock() / CLOCKS_PER_SEC;
        cpuMatMul(h_A, h_B, h_C, n);
        double seq_time = (double)clock() / CLOCKS_PER_SEC - cpu_start;

        float *d_A, *d_B, *d_C;
        CHECK(cudaMalloc(&d_A, tot * sizeof(float)));
        CHECK(cudaMalloc(&d_B, tot * sizeof(float)));
        CHECK(cudaMalloc(&d_C, tot * sizeof(float)));
        CHECK(cudaMemcpy(d_A, h_A, tot * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_B, h_B, tot * sizeof(float), cudaMemcpyHostToDevice));

        dim3 threads(TILE, TILE);
        dim3 blocks((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
        cudaEvent_t e1, e2;
        float ms;
        cudaEventCreate(&e1);
        cudaEventCreate(&e2);

        cudaMemset(d_C, 0, tot * sizeof(float));
        cudaEventRecord(e1);
        matMulGlobal<<<blocks, threads>>>(d_A, d_B, d_C, n);
        cudaEventRecord(e2);
        cudaEventSynchronize(e2);
        cudaEventElapsedTime(&ms, e1, e2);
        double par_global = ms / 1000.0;

        cudaMemset(d_C, 0, tot * sizeof(float));
        cudaEventRecord(e1);
        matMulShared<<<blocks, threads>>>(d_A, d_B, d_C, n);
        cudaEventRecord(e2);
        cudaEventSynchronize(e2);
        cudaEventElapsedTime(&ms, e1, e2);
        double par_shared = ms / 1000.0;

        double sp_global  = seq_time / par_global;
        double sp_shared  = seq_time / par_shared;
        double efficiency = sp_shared / smCount;

        cout << n << "\t" << seq_time << "\t\t" << par_global << "\t\t" << par_shared << "\t\t"
             << sp_global << "\t\t" << sp_shared << "\t\t" << efficiency << endl;

        csv << n << "," << seq_time << "," << par_global << "," << par_shared << ","
            << sp_global << "," << sp_shared << "," << efficiency << "\n";

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        cudaEventDestroy(e1); cudaEventDestroy(e2);
        delete[] h_A; delete[] h_B; delete[] h_C;
    }

    csv.close();
    cout << "\nSaved to 4_matrix_mul.csv" << endl;
    return 0;
}
