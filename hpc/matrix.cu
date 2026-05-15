// Assignment 4: Vector Addition and Matrix Multiplication using CUDA
// Compile: nvcc 4_vector_matrix.cu -o matrix
// Run:     ./matrix

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
        cout << "CUDA error: " << cudaGetErrorString(e) << " at line " << __LINE__ << endl; \
        exit(1); \
    } \
}



__global__ void matMulGlobal(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float s = 0;
        for (int k = 0; k < n; k++)
            s += A[row * n + k] * B[k * n + col];
        C[row * n + col] = s;
    }
}

__global__ void matMulShared(float *A, float *B, float *C, int n) {
    __shared__ float tA[TILE][TILE];
    __shared__ float tB[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float s = 0;
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
            s += tA[threadIdx.y][k] * tB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < n && col < n)
        C[row * n + col] = s;
}

void cpuMatMul(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int k = 0; k < n; k++)
                s += A[i * n + k] * B[k * n + j];
            C[i * n + j] = s;
        }
    }
}

int main() {
  

    int smCount;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);

    
    ofstream csv("matrix.csv");
    csv << "type,size,threads_per_block,seq_time,par_time,speedup,efficiency\n";


    
        
  
    int mat_sizes[] = {8, 16, 32, 64, 128, 256, 512};

    cout << "\n=== Matrix Multiplication (NxN) ===" << endl;
    cout << "N\tSeq(s)\t\tGlobal(s)\tShared(s)\tSp(Gl)\tSp(Sh)\tEfficiency" << endl;

    for (int t = 0; t < 7; t++) {
        int n = mat_sizes[t];
        long long tot = (long long)n * n;

        float *h_A = new float[tot];
        float *h_B = new float[tot];
        float *h_C = new float[tot]();
        for (long long i = 0; i < tot; i++) {
            h_A[i] = rand() % 10;
            h_B[i] = rand() % 10;
        }

        double seqTime = 0;
        if (n <= 512) {
            double cpuS = (double)clock() / CLOCKS_PER_SEC;
            cpuMatMul(h_A, h_B, h_C, n);
            seqTime = (double)clock() / CLOCKS_PER_SEC - cpuS;
        }

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
        double parGl = ms / 1000.0;

        cudaMemset(d_C, 0, tot * sizeof(float));
        cudaEventRecord(e1);
        matMulShared<<<blocks, threads>>>(d_A, d_B, d_C, n);
        cudaEventRecord(e2);
        cudaEventSynchronize(e2);
        cudaEventElapsedTime(&ms, e1, e2);
        double parSh = ms / 1000.0;

        double spGl = (seqTime > 0) ? seqTime / parGl : 0;
        double spSh = (seqTime > 0) ? seqTime / parSh : 0;
        double eff  = (spSh > 0)    ? spSh / smCount  : 0;

        cout << n << "\t" << seqTime << "\t\t" << parGl << "\t\t" << parSh << "\t\t"
             << spGl << "\t" << spSh << "\t" << eff;
        if (n > 512) cout << "\t(CPU skipped)";
        cout << endl;

        csv << "matrix_mul," << n << ",256," << seqTime << "," << parSh << ","
            << spSh << "," << eff << "\n";

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaEventDestroy(e1);
        cudaEventDestroy(e2);
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
    }

    csv.close();
    cout << "\nSaved to matrix.csv" << endl;
    return 0;
}
