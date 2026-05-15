// Parallel Merge Sort using OpenMP
// Compile: g++ -fopenmp mergesort.cpp -o mergesort
// Run:     ./mergesort

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <omp.h>
using namespace std;

void merge(int *a, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    int *L = new int[n1];
    int *R = new int[n2];
    for (int i = 0; i < n1; i++) L[i] = a[left + i];
    for (int i = 0; i < n2; i++) R[i] = a[mid + 1 + i];
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) a[k++] = L[i++];
        else               a[k++] = R[j++];
    }
    while (i < n1) a[k++] = L[i++];
    while (j < n2) a[k++] = R[j++];
    delete[] L;
    delete[] R;
}

void seqMerge(int *a, int left, int right) {
    if (left >= right) return;
    int mid = (left + right) / 2;
    seqMerge(a, left, mid);
    seqMerge(a, mid + 1, right);
    merge(a, left, mid, right);
}

void parMerge(int *a, int left, int right, int depth) {
    if (left >= right) return;
    int mid = (left + right) / 2;
    if (depth > 0) {
        #pragma omp task
        parMerge(a, left, mid, depth - 1);
        #pragma omp task
        parMerge(a, mid + 1, right, depth - 1);
        #pragma omp taskwait
    } else {
        seqMerge(a, left, mid);
        seqMerge(a, mid + 1, right);
    }
    merge(a, left, mid, right);
}

int main() {
    int sizes[] = {1000, 5000, 10000, 30000, 50000};
    int n = 5;
    int threads = omp_get_max_threads();

    cout << "Neeti Kurulkar BE A 41038" << endl;
    cout << "Threads: " << threads << endl << endl;
    cout << "Size\tSeq\t\tPar\t\tSpeedup\t\tEfficiency" << endl;

    ofstream csv("mergesort.csv");
    csv << "size,seq_time,par_time,speedup,efficiency\n";

    for (int t = 0; t < n; t++) {
        int sz = sizes[t];

        int *base = new int[sz];
        srand(42 + t);
        for (int i = 0; i < sz; i++)
            base[i] = rand() % 100000;

        int *a1 = new int[sz];
        int *a2 = new int[sz];
        memcpy(a1, base, sz * sizeof(int));
        memcpy(a2, base, sz * sizeof(int));

        double start;

        start = omp_get_wtime();
        seqMerge(a1, 0, sz - 1);
        double seq_time = omp_get_wtime() - start;

        start = omp_get_wtime();
        #pragma omp parallel
        {
            #pragma omp single
            parMerge(a2, 0, sz - 1, 4);
        }
        double par_time = omp_get_wtime() - start;

        double speedup    = seq_time / par_time;
        double efficiency = speedup / threads;

        cout << sz << "\t" << seq_time << "\t\t" << par_time << "\t\t"
             << speedup << "\t\t" << efficiency << endl;

        csv << sz << "," << seq_time << "," << par_time << ","
            << speedup << "," << efficiency << "\n";

        delete[] base;
        delete[] a1;
        delete[] a2;
    }

    csv.close();
    cout << "\nSaved to mergesort.csv" << endl;
    return 0;
}
