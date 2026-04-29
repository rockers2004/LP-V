#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <omp.h>
#include <stack>
using namespace std;

// ---------------- GRAPH CLASS ----------------
class Graph {
public:
    int V;
    vector<vector<int>> adj;

    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // ---------------- SEQUENTIAL BFS ----------------
    void seqBFS(int start) {
        vector<char> visited(V, 0);
        queue<int> q;

        visited[start] = 1;
        q.push(start);

        while (!q.empty()) {
            int node = q.front();
            q.pop();

            for (int n : adj[node]) {
                if (!visited[n]) {
                    visited[n] = 1;
                    q.push(n);
                }
            }
        }
    }

    // ---------------- OPTIMIZED PARALLEL BFS ----------------
    void parBFS(int start) {
        vector<char> visited(V, 0);
        vector<int> frontier, next_frontier;

        visited[start] = 1;
        frontier.push_back(start);

        while (!frontier.empty()) {

            next_frontier.clear();

            #pragma omp parallel
            {
                vector<int> local_next;

                // schedule(guided) allows dynamic load balancing with decreasing chunk sizes
                #pragma omp for schedule(guided, 1)
                for (size_t i = 0; i < frontier.size(); i++) {
                    int node = frontier[i];

                    for (int n : adj[node]) {

                        // LOCK-FREE VISITED CHECK using atomic CAS
                        if (!visited[n]) {
                            if (__sync_bool_compare_and_swap(&visited[n], 0, 1)) {
                                local_next.push_back(n);
                            }
                        }
                    }
                }

                // Merge once per thread to reduce contention on critical section
                #pragma omp critical
                next_frontier.insert(next_frontier.end(),
                                     local_next.begin(),
                                     local_next.end());
            }

            frontier.swap(next_frontier);
        }
    }

    // ---------------- SEQUENTIAL DFS ----------------
    void seqDFS(int start) {
        vector<char> visited(V, 0);
        stack<int> st;

        st.push(start);

        while (!st.empty()) {
            int node = st.top();
            st.pop();

            if (visited[node]) continue;
            visited[node] = 1;

            for (int n : adj[node]) {
                if (!visited[n])
                    st.push(n);
            }
        }
    }

    // ---------------- OPTIMIZED PARALLEL DFS ----------------
    void parDFS(int start) {
        vector<char> visited(V, 0);
        vector<int> current;
        current.push_back(start);

        while (!current.empty()) {

            vector<int> next;

            #pragma omp parallel
            {
                vector<int> local_next;

                // schedule(guided, 1) provides dynamic load balancing with small chunks
                #pragma omp for schedule(guided, 1)
                for (size_t i = 0; i < current.size(); i++) {
                    int node = current[i];

                    if (!visited[node]) {
                        if (__sync_bool_compare_and_swap(&visited[node], 0, 1)) {

                            for (int n : adj[node]) {
                                if (!visited[n])
                                    local_next.push_back(n);
                            }
                        }
                    }
                }

                #pragma omp critical
                next.insert(next.end(),
                            local_next.begin(),
                            local_next.end());
            }

            current.swap(next);
        }
    }
};

// ---------------- MAIN ----------------
int main() {

    ofstream file("result.txt");
    file << "N,SEQ_TIME,PAR_TIME,SPEEDUP,EFFICIENCY\n";

    // Let OpenMP automatically determine optimal number of threads
    // based on the system (CPU cores, environment variables, etc.)
    // You can also set OMP_NUM_THREADS environment variable to override
    int max_threads = omp_get_max_threads();
    cout << "Using " << max_threads << " threads automatically detected by OpenMP\n\n";

    for (int N = 100; N <= 200000; N += 10000) {

        Graph g(N);

        // Make graph denser (important!)
        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < min(N, i + 50); j++) {
                g.addEdge(i, j);
            }
        }

        double start, end;

        // -------- SEQUENTIAL --------
        start = omp_get_wtime();
        g.seqBFS(0);
        g.seqDFS(0);
        end = omp_get_wtime();
        double seqTime = end - start;

        // -------- PARALLEL --------
        start = omp_get_wtime();
        g.parBFS(0);
        g.parDFS(0);
        end = omp_get_wtime();
        double parTime = end - start;

        // -------- METRICS --------
        double speedup = seqTime / parTime;
        int cores = omp_get_max_threads();
        double efficiency = speedup / cores;

        cout << "N=" << N
             << " Seq=" << seqTime
             << " Par=" << parTime
             << " Speedup=" << speedup
             << " Efficiency=" << efficiency << endl;

        file << N << ","
             << seqTime << ","
             << parTime << ","
             << speedup << ","
             << efficiency << "\n";
    }

    file.close();
    cout << "\nResults saved in result.txt\n";

    return 0;
}
