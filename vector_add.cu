// Vector Addition using CUDA
// Compile: nvcc 4_vector_add.cu -o 4_vector_add
// Run:     ./4_vector_add

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

#define CHECK(c) { \
    cudaError_t e = (c); \
    if (e != cudaSuccess) { \
        cout << "CUDA error: " << cudaGetErrorString(e) << endl; \
        exit(1); \
    } \
}

__global__ void vecAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

void cpuVecAdd(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

int main() {
    cout << "Neeti Kurulkar BE A 41038" << endl << endl;

    int smCount;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);

    float *tmp;
    cudaMalloc(&tmp, sizeof(float));
    vecAdd<<<1, 1>>>(tmp, tmp, tmp, 1);
    cudaDeviceSynchronize();
    cudaFree(tmp);

    // Part 1: Varying input sizes (fixed 256 threads/block)
    int sizes[] = {1000, 5000, 10000, 15000, 20000, 50000};
    int num_sizes = 6;

    ofstream csv1("4_vector_add.csv");
    csv1 << "size,seq_time,par_time,speedup,efficiency\n";

    cout << "=== Varying Input Size (256 threads/block) ===" << endl;
    cout << "Size\t\tSeq(s)\t\tPar(s)\t\tSpeedup\t\tEfficiency" << endl;

    for (int t = 0; t < num_sizes; t++) {
        int n = sizes[t];

        float *h_a = new float[n];
        float *h_b = new float[n];
        float *h_c = new float[n];
        for (int i = 0; i < n; i++) {
            h_a[i] = rand() % 100;
            h_b[i] = rand() % 100;
        }

        double cpu_start = (double)clock() / CLOCKS_PER_SEC;
        cpuVecAdd(h_a, h_b, h_c, n);
        double seq_time = (double)clock() / CLOCKS_PER_SEC - cpu_start;

        float *d_a, *d_b, *d_c;
        CHECK(cudaMalloc(&d_a, n * sizeof(float)));
        CHECK(cudaMalloc(&d_b, n * sizeof(float)));
        CHECK(cudaMalloc(&d_c, n * sizeof(float)));
        CHECK(cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice));

        int blocks = (n + 255) / 256;
        cudaEvent_t e1, e2;
        float ms;
        cudaEventCreate(&e1);
        cudaEventCreate(&e2);
        cudaEventRecord(e1);
        vecAdd<<<blocks, 256>>>(d_a, d_b, d_c, n);
        cudaEventRecord(e2);
        cudaEventSynchronize(e2);
        cudaEventElapsedTime(&ms, e1, e2);
        double par_time = ms / 1000.0;

        double speedup    = seq_time / par_time;
        double efficiency = speedup / smCount;

        cout << n << "\t\t" << seq_time << "\t\t" << par_time << "\t\t"
             << speedup << "\t\t" << efficiency << endl;
        csv1 << n << "," << seq_time << "," << par_time << ","
             << speedup << "," << efficiency << "\n";

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        cudaEventDestroy(e1); cudaEventDestroy(e2);
        delete[] h_a; delete[] h_b; delete[] h_c;
    }
    csv1.close();

    // Part 2: Varying threads/block (fixed n = 10M)
    int fixed_n = 10000000;
    int thread_counts[] = {32, 64, 128, 256, 512, 1024};

    float *h_a = new float[fixed_n];
    float *h_b = new float[fixed_n];
    float *h_c = new float[fixed_n];
    for (int i = 0; i < fixed_n; i++) {
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }

    double cpu_start = (double)clock() / CLOCKS_PER_SEC;
    cpuVecAdd(h_a, h_b, h_c, fixed_n);
    double seq_fixed = (double)clock() / CLOCKS_PER_SEC - cpu_start;

    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, fixed_n * sizeof(float)));
    CHECK(cudaMalloc(&d_b, fixed_n * sizeof(float)));
    CHECK(cudaMalloc(&d_c, fixed_n * sizeof(float)));
    CHECK(cudaMemcpy(d_a, h_a, fixed_n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, fixed_n * sizeof(float), cudaMemcpyHostToDevice));

    ofstream csv2("4_threads.csv");
    csv2 << "threads_per_block,par_time,speedup,efficiency\n";

    cout << "\n=== Varying Threads/Block (n = 10M) ===" << endl;
    cout << "Threads/Block\tPar(s)\t\tSpeedup\t\tEfficiency" << endl;

    for (int ti = 0; ti < 6; ti++) {
        int tc = thread_counts[ti];
        int blocks = (fixed_n + tc - 1) / tc;
        cudaEvent_t e1, e2;
        float ms;
        cudaEventCreate(&e1);
        cudaEventCreate(&e2);
        cudaEventRecord(e1);
        vecAdd<<<blocks, tc>>>(d_a, d_b, d_c, fixed_n);
        cudaEventRecord(e2);
        cudaEventSynchronize(e2);
        cudaEventElapsedTime(&ms, e1, e2);
        double par_time = ms / 1000.0;
        double speedup    = seq_fixed / par_time;
        double efficiency = speedup / smCount;

        cout << tc << "\t\t" << par_time << "\t\t" << speedup << "\t\t" << efficiency << endl;
        csv2 << tc << "," << par_time << "," << speedup << "," << efficiency << "\n";

        cudaEventDestroy(e1); cudaEventDestroy(e2);
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c;

    csv2.close();
    cout << "\nSaved to 4_vector_add.csv and 4_threads.csv" << endl;
    return 0;
}
