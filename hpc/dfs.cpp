// Compile: g++ -fopenmp dfs.cpp -o dfs
// Run:     ./dfs

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

void seqDFS(int u, int *vis) {
    vis[u] = 1;
    for (int i = 0; i < V; i++) {
        if (adj[u][i] && !vis[i])
            seqDFS(i, vis);
    }
}

void parDFS(int u, int *vis) {
    for (int i = 0; i < V; i++) {
        if (adj[u][i] && !vis[i]) {
            int go = 0;
            #pragma omp critical
            {
                if (!vis[i]) {
                    vis[i] = 1;
                    go = 1;
                }
            }
            if (go) {
                #pragma omp task
                parDFS(i, vis);
            }
        }
    }
    #pragma omp taskwait
}

int main() {
    // Hardcoded sizes: small to large so chain graph lines cross
    int sizes[] = {100, 500, 1000, 2000, 5000};
    int n = 5;
    int threads = omp_get_max_threads();

    
    cout << "Threads: " << threads << endl << endl;
    cout << "Size\tSeqDFS\t\tParDFS\t\tSpDFS\tEfficiency" << endl;

    ofstream csv("dfs.csv");
    csv << "size,seq_dfs,par_dfs,speedup_dfs,efficiency\n";

    for (int t = 0; t < n; t++) {
        initGraph(sizes[t]);
        int max_e = sizes[t] * (sizes[t] - 1) / 2;
        int m = (sizes[t] * 3 < max_e) ? sizes[t] * 3 : max_e;
        genEdges(m);

        double start;

        int *v1 = new int[sizes[t]]();
        start = omp_get_wtime();
        seqDFS(0, v1);
        double seqDfs = omp_get_wtime() - start;
        delete[] v1;

        int *v2 = new int[sizes[t]]();
        v2[0] = 1;
        start = omp_get_wtime();
        #pragma omp parallel
        {
            #pragma omp single
            parDFS(0, v2);
        }
        double parDfs = omp_get_wtime() - start;
        delete[] v2;

        
        double spD = seqDfs / parDfs;
        double eff =  spD / threads;

        cout << sizes[t] << "\t"
             << seqDfs << "\t" << parDfs << "\t" << spD << "\t" << eff << endl;

        csv << sizes[t] << ","
            << seqDfs << "," << parDfs << "," << spD << "," << eff << "\n";

        freeGraph();
    }

    csv.close();
    cout << "\nSaved to dfs.csv" << endl;
    return 0;
}
