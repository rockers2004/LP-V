// Parallel Bubble Sort (Odd-Even) using OpenMP
// Compile: g++ -fopenmp bubblesort.cpp -o bubblesort
// Run:     ./bubblesort

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <omp.h>
using namespace std;

void seqBubble(int *a, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1 - i; j++) {
            if (a[j] > a[j + 1]) {
                int tmp = a[j];
                a[j] = a[j + 1];
                a[j + 1] = tmp;
            }
        }
    }
}

void parBubble(int *a, int n) {
    #pragma omp parallel shared(a)
    {
        for (int phase = 0; phase < n; phase++) {
            #pragma omp for
            for (int j = phase % 2; j < n - 1; j += 2) {
                if (a[j] > a[j + 1]) {
                    int tmp = a[j];
                    a[j] = a[j + 1];
                    a[j + 1] = tmp;
                }
            }
        }
    }
}

int main() {
    int sizes[] = {1000, 5000, 10000, 30000, 50000};
    int n = 5;
    int threads = omp_get_max_threads();

    cout << "Neeti Kurulkar BE A 41038" << endl;
    cout << "Threads: " << threads << endl << endl;
    cout << "Size\tSeq\t\tPar\t\tSpeedup\t\tEfficiency" << endl;

    ofstream csv("2_bubble_sort.csv");
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
        seqBubble(a1, sz);
        double seq_time = omp_get_wtime() - start;

        start = omp_get_wtime();
        parBubble(a2, sz);
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
    cout << "\nSaved to 2_bubble_sort.csv" << endl;
    return 0;
}