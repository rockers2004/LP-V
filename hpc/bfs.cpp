// Compile: g++ -fopenmp 1_bfs.cpp -o 1_bfs
// Run:     ./1_bfs

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <omp.h>
using namespace std;

int V, **adj;

void initGraph(int v) {
    V = v;
    adj = new int*[V];
    for (int i = 0; i < V; i++)
        adj[i] = new int[V]();
}

void freeGraph() {
    for (int i = 0; i < V; i++)
        delete[] adj[i];
    delete[] adj;
}

void addEdge(int u, int v) {
    adj[u][v] = 1;
    adj[v][u] = 1;
}

void genEdges(int m) {
    srand(42);
    int e = 0;
    while (e < m) {
        int u = rand() % V;
        int v = rand() % V;
        if (u != v && !adj[u][v]) {
            addEdge(u, v);
            e++;
        }
    }
}

void seqBFS(int s) {
    int *vis = new int[V]();
    int *q   = new int[V];
    int f = 0, r = 0;
    vis[s] = 1;
    q[r++] = s;
    while (f < r) {
        int u = q[f++];
        for (int i = 0; i < V; i++) {
            if (adj[u][i] && !vis[i]) {
                vis[i] = 1;
                q[r++] = i;
            }
        }
    }
    delete[] vis;
    delete[] q;
}

void parBFS(int s) {
    int *vis = new int[V]();
    int *q   = new int[V];
    int f = 0, r = 0;
    vis[s] = 1;
    q[r++] = s;
    while (f < r) {
        int lf = f, lr = r;
        #pragma omp parallel for shared(vis, q, r)
        for (int i = lf; i < lr; i++) {
            int u = q[i];
            for (int j = 0; j < V; j++) {
                if (adj[u][j] && !vis[j]) {
                    #pragma omp critical
                    {
                        if (!vis[j]) {
                            vis[j] = 1;
                            q[r++] = j;
                        }
                    }
                }
            }
        }
        f = lr;
    }
    delete[] vis;
    delete[] q;
}

int main() {
    // Hardcoded sizes: small to large so chain graph lines cross
    int sizes[] = {100, 500, 1000, 2000, 5000};
    int n = 5;
    int threads = omp_get_max_threads();

    
    cout << "Threads: " << threads << endl << endl;
    cout << "Size\tSeqBFS\t\tParBFS\t\tSpBFS\tEfficiency" << endl;

    ofstream csv("1_bfs.csv");
    csv << "size,seq_bfs,par_bfs,speedup_bfs,efficiency\n";

    for (int t = 0; t < n; t++) {
        initGraph(sizes[t]);
        int max_e = sizes[t] * (sizes[t] - 1) / 2;
        int m = (sizes[t] * 3 < max_e) ? sizes[t] * 3 : max_e;
        genEdges(m);

        double start;

        start = omp_get_wtime();
        seqBFS(0);
        double seqBfs = omp_get_wtime() - start;

        start = omp_get_wtime();
        parBFS(0);
        double parBfs = omp_get_wtime() - start;
	double spB = seqBfs / parBfs;
	double eff = spB  / threads;

        cout << sizes[t] << "\t"
             << seqBfs << "\t" << parBfs << "\t" << spB << "\t"
             << seqDfs << "\t" << parDfs << "\t" << spD << "\t" << eff << endl;

        csv << sizes[t] << ","
            << seqBfs << "," << parBfs << "," << spB << ","
              << eff << "\n";

        freeGraph();
    }

    csv.close();
    cout << "\nSaved to 1_bfs.csv" << endl;
    return 0;
}
