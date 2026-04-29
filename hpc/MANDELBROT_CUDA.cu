#include <iostream>
#include <cuda.h>
#include <chrono>
#include <fstream>

using namespace std;

// ---------------- GPU KERNEL ----------------
__global__ void mandelbrotKernel(int *image, int width, int height, int max_iter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int < width && y < height) {

        float zx = 0.0f, zy = 0.0f;
        float cx = (x - width / 2.0f) * 4.0f / width;
        float cy = (y - height / 2.0f) * 4.0f / height;

        int count = 0;

        while (zx * zx + zy * zy < 4.0f && count < max_iter) {
            float temp = zx * zx - zy * zy + cx;
            zy = 2.0f * zx * zy + cy;
            zx = temp;
            count++;
        }

        image[y * width + x] = count;
    } y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x
}

// ---------------- CPU VERSION ----------------
void mandelbrotCPU(int *image, int width, int height, int max_iter) {
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {

            float zx = 0.0f, zy = 0.0f;
            float cx = (x - width / 2.0f) * 4.0f / width;
            float cy = (y - height / 2.0f) * 4.0f / height;

            int count = 0;

            while (zx * zx + zy * zy < 4.0f && count < max_iter) {
                float temp = zx * zx - zy * zy + cx;
                zy = 2.0f * zx * zy + cy;
                zx = temp;
                count++;
            }

            image[y * width + x] = count;
        }
    }
}

// ---------------- MAIN ----------------
int main() {

    ofstream file("result.txt");
    file << "SIZE,CPU_TIME,GPU_TIME,SPEEDUP,EFFICIENCY\n";

    int sizes[] = {256, 512, 1024, 2048, 3072};
    int max_iter = 100;

    for (int size : sizes) {

        int width = size;
        int height = size;
        int total = width * height;

        int *h_image = new int[total];
        int *d_image;

        cudaMalloc(&d_image, total * sizeof(int));

        // -------- CPU --------
        auto start = chrono::high_resolution_clock::now();
        mandelbrotCPU(h_image, width, height, max_iter);
        auto end = chrono::high_resolution_clock::now();
        double cpu_time = chrono::duration<double, milli>(end - start).count();

        // -------- GPU --------
        dim3 threads(16, 16);
        dim3 blocks((width + 15)/16, (height + 15)/16);

        start = chrono::high_resolution_clock::now();
        mandelbrotKernel<<<blocks, threads>>>(d_image, width, height, max_iter);
        cudaDeviceSynchronize();
        end = chrono::high_resolution_clock::now();
        double gpu_time = chrono::duration<double, milli>(end - start).count();

        double speedup = cpu_time / gpu_time;
        int cores = 2560; // T4 GPU
        double efficiency = speedup / cores;

        cout << "Size=" << size
             << " CPU=" << cpu_time
             << " GPU=" << gpu_time
             << " Speedup=" << speedup << endl;

        file << size << ","
             << cpu_time << ","
             << gpu_time << ","
             << speedup << ","
             << efficiency << "\n";

        cudaFree(d_image);
        delete[] h_image;
    }

    file.close();
    cout << "\nSaved results to result.txt\n";

    return 0;
}
