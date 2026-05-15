// Compile: g++ -fopenmp merge_sorting.cpp -o merge_sorting
// Run:     ./merge_sorting

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <omp.h>
using namespace std;




void merge(int *a, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;
    int *L = new int[n1];
    int *R = new int[n2];
    for (int i = 0; i < n1; i++) L[i] = a[l + i];
    for (int i = 0; i < n2; i++) R[i] = a[m + 1 + i];
    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) a[k++] = L[i++];
        else               a[k++] = R[j++];
    }
    while (i < n1) a[k++] = L[i++];
    while (j < n2) a[k++] = R[j++];
    delete[] L;
    delete[] R;
}

void seqMerge(int *a, int l, int r) {
    if (l >= r) return;
    int m = (l + r) / 2;
    seqMerge(a, l, m);
    seqMerge(a, m + 1, r);
    merge(a, l, m, r);
}

void parMerge(int *a, int l, int r, int depth) {
    if (l >= r) return;
    int m = (l + r) / 2;
    if (depth > 0) {
        #pragma omp task
        parMerge(a, l, m, depth - 1);
        #pragma omp task
        parMerge(a, m + 1, r, depth - 1);
        #pragma omp taskwait
    } else {
        seqMerge(a, l, m);
        seqMerge(a, m + 1, r);
    }
    merge(a, l, m, r);
}

int main() {
    // Hardcoded sizes: small to large so chain graph lines cross
    int sizes[] = {1000, 5000, 10000, 30000, 50000};
    int n = 5;
    int threads = omp_get_max_threads();

    
    cout << "Threads: " << threads << endl << endl;
    cout << "Size\tSeqMerge\tParMerge\tSpMerge\tEfficiency" << endl;

    ofstream csv("merge_sorting.csv");
    csv << "size,seq_merge,par_merge,speedup_merge,efficiency\n";

    for (int t = 0; t < n; t++) {
        int sz = sizes[t];

        int *base = new int[sz];
        srand(42 + t);
        for (int i = 0; i < sz; i++)
            base[i] = rand() % 100000;

       
        int *a3 = new int[sz];
        int *a4 = new int[sz];
       
        memcpy(a3, base, sz * sizeof(int));
        memcpy(a4, base, sz * sizeof(int));

        double start;

        

        start = omp_get_wtime();
        seqMerge(a3, 0, sz - 1);
        double seqMer = omp_get_wtime() - start;

        start = omp_get_wtime();
        #pragma omp parallel
        {
            #pragma omp single
            parMerge(a4, 0, sz - 1, 4);
        }
        double parMer = omp_get_wtime() - start;

        double spM = seqMer / parMer;
        double eff =  spM / threads;

        cout << sz << "\t"
             
             << seqMer << "\t" << parMer << "\t" << spM << "\t" << eff << endl;

        csv << sz << ","
            
            << seqMer << "," << parMer << "," << spM << "," << eff << "\n";

        delete[] base;
        
        delete[] a3;
        delete[] a4;
    }

    csv.close();
    cout << "\nSaved to merge_sorting.csv" << endl;
    return 0;
}
