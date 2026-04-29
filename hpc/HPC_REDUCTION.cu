#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>

__global__ void sum_reduction(float* d_out, float* d_in, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? d_in[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        d_out[blockIdx.x] = sdata[0];
}

// -------- FULL REDUCTION --------
float gpu_sum(float* d_in, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    float *d_out;
    cudaMalloc(&d_out, blocks * sizeof(float));

    sum_reduction<<<blocks, threads, threads * sizeof(float)>>>(d_out, d_in, n);
    cudaDeviceSynchronize();

    // Copy partial sums to CPU
    std::vector<float> h_out(blocks);
    cudaMemcpy(h_out.data(), d_out, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float final_sum = 0;
    for (int i = 0; i < blocks; i++)
        final_sum += h_out[i];

    cudaFree(d_out);
    return final_sum;
}

// -------- TEST FUNCTION --------
void run_test(int n, std::ofstream& outfile) {
    size_t bytes = n * sizeof(float);
    std::vector<float> h_in(n, 1.0f);

    float *d_in;
    cudaMalloc(&d_in, bytes);
    cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

    // -------- SERIAL --------
    float h_sum_cpu = 0;
    auto start_s = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++)
        h_sum_cpu += h_in[i];
    auto end_s = std::chrono::high_resolution_clock::now();
    double time_s = std::chrono::duration<double, std::milli>(end_s - start_s).count();

    // -------- PARALLEL --------
    auto start_p = std::chrono::high_resolution_clock::now();
    float h_sum_gpu = gpu_sum(d_in, n);
    auto end_p = std::chrono::high_resolution_clock::now();
    double time_p = std::chrono::duration<double, std::milli>(end_p - start_p).count();

    int cores = 2560; 

    double speedup = time_s / time_p;
    double efficiency = speedup / cores;

    std::cout << "N=" << n
              << " CPU=" << h_sum_cpu
              << " GPU=" << h_sum_gpu
              << " Speedup=" << speedup << std::endl;

    outfile << n << ","
            << time_s << ","
            << time_p << ","
            << speedup << ","
            << efficiency << std::endl;

    cudaFree(d_in);
}

// -------- MAIN --------
int main() {
    std::ofstream outfile("result.txt");
    outfile << "N,SERIAL,PARALLEL,SPEEDUP,EFFICIENCY\n";

    int sizes[] = {1000, 10000, 100000, 1000000, 5000000};

    for (int n : sizes)
        run_test(n, outfile);

    outfile.close();

    return 0;
}
