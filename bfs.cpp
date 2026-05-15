// Parallel BFS using OpenMP
// Compile: g++ -fopenmp bfs.cpp -o bfs
// Run:     ./bfs

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

void seqBFS(int start) {
    int *visited = new int[V]();
    int *queue   = new int[V];
    int front = 0, rear = 0;
    visited[start] = 1;
    queue[rear++] = start;
    while (front < rear) {
        int u = queue[front++];
        for (int i = 0; i < V; i++) {
            if (adj[u][i] && !visited[i]) {
                visited[i] = 1;
                queue[rear++] = i;
            }
        }
    }
    delete[] visited;
    delete[] queue;
}

void parBFS(int start) {
    int *visited = new int[V]();
    int *queue   = new int[V];
    int front = 0, rear = 0;
    visited[start] = 1;
    queue[rear++] = start;
    while (front < rear) {
        int lf = front, lr = rear;
        #pragma omp parallel for shared(visited, queue, rear)
        for (int i = lf; i < lr; i++) {
            int u = queue[i];
            for (int j = 0; j < V; j++) {
                if (adj[u][j] && !visited[j]) {
                    #pragma omp critical
                    {
                        if (!visited[j]) {
                            visited[j] = 1;
                            queue[rear++] = j;
                        }
                    }
                }
            }
        }
        front = lr;
    }
    delete[] visited;
    delete[] queue;
}

int main() {
    int sizes[] = {100, 500, 1000, 2000, 5000};
    int n = 5;
    int threads = omp_get_max_threads();

    cout << "Neeti Kurulkar BE A 41038" << endl;
    cout << "Threads: " << threads << endl << endl;
    cout << "Size\tSeq BFS\t\tPar BFS\t\tSpeedup\t\tEfficiency" << endl;

    ofstream csv("bfs.csv");
    csv << "size,seq_time,par_time,speedup,efficiency\n";

    for (int t = 0; t < n; t++) {
        initGraph(sizes[t]);
        int max_e = sizes[t] * (sizes[t] - 1) / 2;
        int m = (sizes[t] * 3 < max_e) ? sizes[t] * 3 : max_e;
        genEdges(m);

        double start;

        start = omp_get_wtime();
        seqBFS(0);
        double seq_time = omp_get_wtime() - start;

        start = omp_get_wtime();
        parBFS(0);
        double par_time = omp_get_wtime() - start;

        double speedup    = seq_time / par_time;
        double efficiency = speedup / threads;

        cout << sizes[t] << "\t" << seq_time << "\t\t" << par_time << "\t\t"
             << speedup << "\t\t" << efficiency << endl;

        csv << sizes[t] << "," << seq_time << "," << par_time << ","
            << speedup << "," << efficiency << "\n";

        freeGraph();
    }

    csv.close();
    cout << "\nSaved to bfs.csv" << endl;
    return 0;
}