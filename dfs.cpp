// Parallel DFS using OpenMP
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
    int count = 0;
    while (count < m) {
        int u = rand() % V;
        int v = rand() % V;
        if (u != v && !adj[u][v]) {
            addEdge(u, v);
            count++;
        }
    }
}

void seqDFS(int u, int *visited) {
    visited[u] = 1;
    for (int i = 0; i < V; i++) {
        if (adj[u][i] && !visited[i])
            seqDFS(i, visited);
    }
}

void parDFS(int u, int *visited) {
    for (int i = 0; i < V; i++) {
        if (adj[u][i] && !visited[i]) {
            int go = 0;
            #pragma omp critical
            {
                if (!visited[i]) {
                    visited[i] = 1;
                    go = 1;
                }
            }
            if (go) {
                #pragma omp task
                parDFS(i, visited);
            }
        }
    }
    #pragma omp taskwait
}

int main() {
    int sizes[] = {100, 500, 1000, 2000, 5000};
    int n = 5;
    int threads = omp_get_max_threads();

    cout << "Neeti Kurulkar BE A 41038" << endl;
    cout << "Threads: " << threads << endl << endl;
    cout << "Size\tSeq DFS\t\tPar DFS\t\tSpeedup\t\tEfficiency" << endl;

    ofstream csv("dfs.csv");
    csv << "size,seq_time,par_time,speedup,efficiency\n";

    for (int t = 0; t < n; t++) {
        initGraph(sizes[t]);
        int max_e = sizes[t] * (sizes[t] - 1) / 2;
        int m = (sizes[t] * 3 < max_e) ? sizes[t] * 3 : max_e;
        genEdges(m);

        double start;

        int *v1 = new int[sizes[t]]();
        start = omp_get_wtime();
        seqDFS(0, v1);
        double seq_time = omp_get_wtime() - start;
        delete[] v1;

        int *v2 = new int[sizes[t]]();
        v2[0] = 1;
        start = omp_get_wtime();
        #pragma omp parallel
        {
            #pragma omp single
            parDFS(0, v2);
        }
        double par_time = omp_get_wtime() - start;
        delete[] v2;

        double speedup    = seq_time / par_time;
        double efficiency = speedup / threads;

        cout << sizes[t] << "\t" << seq_time << "\t\t" << par_time << "\t\t"
             << speedup << "\t\t" << efficiency << endl;

        csv << sizes[t] << "," << seq_time << "," << par_time << ","
            << speedup << "," << efficiency << "\n";

        freeGraph();
    }

    csv.close();
    cout << "\nSaved to dfs.csv" << endl;
    return 0;
}