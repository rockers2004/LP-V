%%cuda
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

#define CHECK(c) { \
    cudaError_t e = (c); \
    if (e != cudaSuccess) { \
        cout << "CUDA error: " << cudaGetErrorString(e) \
             << " at line " << __LINE__ << endl; \
        exit(1); \
    } \
}

// ── Vector Addition Kernel ───────────────────────────────────────────
__global__ void vecAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

// ── CPU Version ──────────────────────────────────────────────────────
void cpuVecAdd(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

int main() {

    srand(time(0));  // Seed random numbers

    int smCount;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);

    // Warm-up CUDA
    float *tmp;
    cudaMalloc(&tmp, sizeof(float));
    vecAdd<<<1,1>>>(tmp, tmp, tmp, 1);
    cudaDeviceSynchronize();
    cudaFree(tmp);

    // Open CSV file
    ofstream csv("1vector_add.csv");
    if (!csv.is_open()) {
        cout << "Error opening CSV file!" << endl;
        return 1;
    }

    csv << "type,size,threads_per_block,seq_time,par_time,speedup,efficiency\n";

    // ── PART 1: Different sizes ───────────────────────────────────────
    int vec_sizes[] = {1000, 5000, 10000, 15000, 20000, 50000};
    int vec_n = 6;

    cout << "=== Vector Addition (threads/block = 256) ===" << endl;
    cout << "Size\tSeq(s)\tPar(s)\tSpeedup\tEfficiency" << endl;

    for (int t = 0; t < vec_n; t++) {

        int n = vec_sizes[t];

        float *h_a = new float[n];
        float *h_b = new float[n];
        float *h_c = new float[n];

        for (int i = 0; i < n; i++) {
            h_a[i] = rand() % 100;
            h_b[i] = rand() % 100;
        }

        // CPU timing
        double cpuStart = (double)clock() / CLOCKS_PER_SEC;
        cpuVecAdd(h_a, h_b, h_c, n);
        double seqTime = (double)clock() / CLOCKS_PER_SEC - cpuStart;

        // GPU memory
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
        CHECK(cudaGetLastError());
        cudaEventRecord(e2);

        cudaEventSynchronize(e2);
        cudaEventElapsedTime(&ms, e1, e2);

        double parTime = ms / 1000.0;
        double speedup = seqTime / parTime;
        double eff = speedup / smCount;

        cout << n << "\t" << seqTime << "\t" << parTime
             << "\t" << speedup << "\t" << eff << endl;

        csv << "vector_add," << n << ",256," << seqTime << ","
            << parTime << "," << speedup << "," << eff << "\n";
        csv.flush();

        // Free memory
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaEventDestroy(e1);
        cudaEventDestroy(e2);

        delete[] h_a;
        delete[] h_b;
        delete[] h_c;
    }

    // ── PART 2: Thread variation ──────────────────────────────────────
    int fixed_n = 10000000;
    int thread_counts[] = {32, 64, 128, 256, 512, 1024};

    float *h_a = new float[fixed_n];
    float *h_b = new float[fixed_n];
    float *h_c = new float[fixed_n];

    for (int i = 0; i < fixed_n; i++) {
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }

    double cpuStart = (double)clock() / CLOCKS_PER_SEC;
    cpuVecAdd(h_a, h_b, h_c, fixed_n);
    double seqFixed = (double)clock() / CLOCKS_PER_SEC - cpuStart;

    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, fixed_n * sizeof(float)));
    CHECK(cudaMalloc(&d_b, fixed_n * sizeof(float)));
    CHECK(cudaMalloc(&d_c, fixed_n * sizeof(float)));

    CHECK(cudaMemcpy(d_a, h_a, fixed_n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, fixed_n * sizeof(float), cudaMemcpyHostToDevice));

    cout << "\n=== Thread Count Effect (n=10M) ===" << endl;
    cout << "Threads\tPar(s)\tSpeedup\tEfficiency" << endl;

    for (int ti = 0; ti < 6; ti++) {

        int tc = thread_counts[ti];
        int blocks = (fixed_n + tc - 1) / tc;

        cudaEvent_t e1, e2;
        float ms;

        cudaEventCreate(&e1);
        cudaEventCreate(&e2);

        cudaEventRecord(e1);
        vecAdd<<<blocks, tc>>>(d_a, d_b, d_c, fixed_n);
        CHECK(cudaGetLastError());
        cudaEventRecord(e2);

        cudaEventSynchronize(e2);
        cudaEventElapsedTime(&ms, e1, e2);

        double parTime = ms / 1000.0;
        double speedup = seqFixed / parTime;
        double eff = speedup / smCount;

        cout << tc << "\t" << parTime
             << "\t" << speedup << "\t" << eff << endl;

        csv << "vec_threads," << fixed_n << "," << tc << ","
            << seqFixed << "," << parTime << ","
            << speedup << "," << eff << "\n";
        csv.flush();

        cudaEventDestroy(e1);
        cudaEventDestroy(e2);
    }

    // Free memory AFTER loop
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    csv.close();

    cout << "\n✅ Results saved to 1vector_add.csv" << endl;

    return 0;
}