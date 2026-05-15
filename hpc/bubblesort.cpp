// Compile: g++ -fopenmp bubble_sorting.cpp -o bubble_sorting
// Run:     ./bubble_sorting

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
                int t = a[j];
                a[j] = a[j + 1];
                a[j + 1] = t;
            }
        }
    }
}

// Odd-Even Phase Sort: threads created once, reused across all phases
void parBubble(int *a, int n) {
    #pragma omp parallel shared(a)
    {
        for (int phase = 0; phase < n; phase++) {
            #pragma omp for
            for (int j = phase % 2; j < n - 1; j += 2) {
                if (a[j] > a[j + 1]) {
                    int t = a[j];
                    a[j] = a[j + 1];
                    a[j + 1] = t;
                }
            }
        }
    }
}

 int l, int r) {
    if (l >= r) return;
    int m = (l + r) / 2;
    seqMerge(a, l, m);
    seqMerge(a, m + 1, r);
    merge(a, l, m, r);
}


int main() {
    // Hardcoded sizes: small to large so chain graph lines cross
    int sizes[] = {1000, 5000, 10000, 30000, 50000};
    int n = 5;
    int threads = omp_get_max_threads();

    
    cout << "Threads: " << threads << endl << endl;
    cout << "Size\tSeqBubble\tParBubble\tSpBubbletEfficiency" << endl;

    ofstream csv("bubble_sorting.csv");
    csv << "size,seq_bubble,par_bubble,speedup_bubble,efficiency\n";

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
        double seqBub = omp_get_wtime() - start;

        start = omp_get_wtime();
        parBubble(a2, sz);
        double parBub = omp_get_wtime() - start;

        

        double spB = seqBub / parBub;
        
        double eff = spB  / threads;

        cout << sz << "\t"
             << seqBub << "\t" << parBub << "\t" << spB << "\t"
            << eff << endl;

        csv << sz << ","
            << seqBub << "," << parBub << "," << spB << ","
             << eff << "\n";

        delete[] base;
        delete[] a1;
        delete[] a2;
        
    }

    csv.close();
    cout << "\nSaved to bubble_sorting.csv" << endl;
    return 0;
}
