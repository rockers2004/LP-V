# High Performance Computing (HPC) Practical + Viva Complete Guide

## Practicals Covered

1. Parallel Bubble Sort using OpenMP
2. Parallel Merge Sort using OpenMP
3. Parallel BFS using OpenMP
4. Parallel DFS using OpenMP
5. CUDA Vector Addition
6. CUDA Matrix Multiplication
7. CUDA Reduction

---

# INDEX

1. Introduction to HPC
2. Parallel Computing Basics
3. OpenMP Complete Theory
4. CUDA Complete Theory
5. Bubble Sort Practical Viva
6. Merge Sort Practical Viva
7. BFS Practical Viva
8. DFS Practical Viva
9. CUDA Vector Addition Viva
10. CUDA Matrix Multiplication Viva
11. CUDA Reduction Viva
12. Frequently Asked External Questions
13. Numerical and Formula Questions
14. Compilation Commands
15. Important Definitions
16. Comparison Tables
17. Last Minute Revision

---

# 1. INTRODUCTION TO HIGH PERFORMANCE COMPUTING

## What is High Performance Computing?

High Performance Computing (HPC) refers to the use of powerful computing resources, parallel processing, and supercomputers to solve large and complex computational problems efficiently.

HPC combines:
- Multiple processors
- Parallel algorithms
- High-speed communication
- Large memory systems

Goal:
- Reduce execution time
- Increase computational performance
- Handle massive datasets

---

## Applications of HPC

### Scientific Simulations
- Climate modeling
- Weather forecasting
- Earthquake prediction

### Artificial Intelligence
- Neural network training
- Deep learning
- Large language models

### Medical Field
- Genome sequencing
- Drug discovery
- MRI analysis

### Engineering
- Fluid simulations
- CAD/CAM
- Finite Element Analysis

### Finance
- Risk analysis
- Algorithmic trading
- Market prediction

---

## Characteristics of HPC

- High speed computation
- Large memory usage
- Parallel execution
- Scalability
- Reliability

---

# 2. PARALLEL COMPUTING BASICS

## What is Parallel Computing?

Parallel computing is a technique where a large problem is divided into smaller independent tasks that execute simultaneously.

Instead of one processor solving the entire problem:
- Multiple processors solve parts together.

---

## Sequential vs Parallel Computing

| Sequential Computing | Parallel Computing |
|---|---|
| One instruction at a time | Multiple instructions simultaneously |
| Single processor | Multiple processors/cores |
| Slower | Faster |
| Simpler | Complex synchronization |
| Less efficient for large data | Highly efficient for large data |

---

## Types of Parallelism

### Data Parallelism
Same operation applied on different data.

Example:
- Vector addition

### Task Parallelism
Different tasks executed simultaneously.

Example:
- Merge sort recursive tasks

---

## Flynn’s Classification

### SISD
Single Instruction Single Data

### SIMD
Single Instruction Multiple Data
GPU architecture follows SIMD.

### MISD
Multiple Instruction Single Data
Rarely used.

### MIMD
Multiple Instruction Multiple Data
Modern multicore CPUs.

---

## Concurrency vs Parallelism

### Concurrency
Multiple tasks make progress together.

### Parallelism
Multiple tasks execute simultaneously.

---

## What is a Thread?

Thread is the smallest unit of execution.

Features:
- Lightweight
- Share same memory
- Faster context switching

---

## Process vs Thread

| Process | Thread |
|---|---|
| Heavyweight | Lightweight |
| Separate memory | Shared memory |
| Slower switching | Faster switching |
| Independent | Dependent |

---

## What is Synchronization?

Synchronization ensures proper coordination between threads.

Purpose:
- Prevent race conditions
- Maintain correctness

---

## What is Race Condition?

Occurs when multiple threads access and modify shared data simultaneously causing unpredictable results.

---

## What is Critical Section?

Part of code accessed by only one thread at a time.

Used to protect shared variables.

---

# 3. PERFORMANCE METRICS

## Speedup

Formula:

Speedup = Sequential Time / Parallel Time

Example:
- Sequential = 10 sec
- Parallel = 2 sec

Speedup = 5

Meaning:
Program became 5 times faster.

---

## Efficiency

Formula:

Efficiency = Speedup / Number of Threads

Example:
- Speedup = 4
- Threads = 8

Efficiency = 0.5 = 50%

---

## Scalability

Ability of system to maintain performance when workload or processors increase.

---

## Throughput

Amount of work completed per unit time.

---

## Latency

Delay before execution begins.

---

## Amdahl’s Law

Shows limitation of parallel systems.

Formula:

Speedup = 1 / ((1-P)+(P/N))

Where:
- P = parallel portion
- N = processors

---

# 4. OPENMP COMPLETE THEORY

## What is OpenMP?

OpenMP is an API for shared-memory parallel programming in C, C++, and Fortran.

It uses:
- Compiler directives
- Runtime libraries
- Environment variables

---

## Advantages of OpenMP

- Easy to use
- Supports multithreading
- Shared memory model
- Incremental parallelization
- Portable

---

## Disadvantages of OpenMP

- Only shared memory systems
- Limited scalability
- Synchronization overhead

---

## Important OpenMP Directives

### #pragma omp parallel
Creates parallel region.

Example:
```cpp
#pragma omp parallel
{
   printf("Hello");
}
```

---

### #pragma omp for
Distributes loop iterations among threads.

---

### #pragma omp critical
Allows only one thread at a time.

---

### #pragma omp task
Creates asynchronous tasks.

Used in:
- DFS
- Merge sort

---

### #pragma omp taskwait
Waits for child tasks.

---

### #pragma omp single
Ensures only one thread executes block.

---

## OpenMP Functions

### omp_get_max_threads()
Returns maximum available threads.

### omp_get_wtime()
Returns wall-clock time.

---

## Shared vs Private Variables

| Shared | Private |
|---|---|
| Accessible to all threads | Each thread gets copy |
| Risk of race conditions | Safe |

---

# 5. CUDA COMPLETE THEORY

## What is CUDA?

CUDA (Compute Unified Device Architecture) is NVIDIA’s parallel computing platform for GPU programming.

Allows thousands of threads to execute simultaneously.

---

## CPU vs GPU

| CPU | GPU |
|---|---|
| Few powerful cores | Thousands of small cores |
| Optimized for sequential tasks | Optimized for parallel tasks |
| Complex logic | Massive parallelism |

---

## CUDA Programming Model

### Host
CPU side.

### Device
GPU side.

---

## What is a Kernel?

Kernel is a function executed on GPU.

Syntax:
```cpp
__global__ void kernel()
```

---

## CUDA Thread Hierarchy

### Thread
Smallest execution unit.

### Block
Group of threads.

### Grid
Collection of blocks.

---

## CUDA Memory Types

### Global Memory
Large but slow.

### Shared Memory
Fast memory shared within block.

### Local Memory
Private to thread.

### Constant Memory
Read-only cached memory.

---

## CUDA Memory Functions

### cudaMalloc()
Allocates GPU memory.

### cudaMemcpy()
Copies data between CPU and GPU.

### cudaFree()
Frees GPU memory.

---

## Thread Index Formula

index = blockIdx.x * blockDim.x + threadIdx.x

---

# 6. PARALLEL BUBBLE SORT USING OPENMP

## Aim

To implement sequential and parallel bubble sort using OpenMP and compare performance.

---

## Theory

Bubble sort repeatedly swaps adjacent elements if they are in wrong order.

Parallel version uses Odd-Even Transposition Sort.

---

## Why Normal Bubble Sort is Difficult to Parallelize?

Dependencies exist between adjacent swaps.

Example:
- Swap at index i affects i+1.

---

## Odd-Even Parallel Bubble Sort

### Even Phase
Compare:
- (0,1)
- (2,3)
- (4,5)

### Odd Phase
Compare:
- (1,2)
- (3,4)
- (5,6)

No conflicts occur.

---

## Important Code Concepts

### phase % 2
Alternates between odd and even phases.

### #pragma omp for
Distributes comparisons among threads.

### shared(a)
All threads access same array.

---

## Time Complexity

Sequential:
O(n²)

Parallel:
Approximately O(n²)

---

## Advantages

- Easy implementation
- Demonstrates synchronization

---

## Disadvantages

- Poor scalability
- Inefficient for large datasets

---

## Important Viva Questions

### Why bubble sort not preferred in real HPC?
Because complexity is O(n²).

### Why odd-even method used?
To avoid simultaneous conflicts.

### What happens without synchronization?
Incorrect sorting due to race conditions.

---

# 7. PARALLEL MERGE SORT USING OPENMP

## Aim

To implement sequential and parallel merge sort using OpenMP tasks.

---

## Theory

Merge sort follows Divide and Conquer.

Steps:
1. Divide array
2. Sort recursively
3. Merge sorted halves

---

## Why Merge Sort is Good for Parallelization?

Left and right halves are independent.

Can execute simultaneously.

---

## Important Functions

### merge()
Combines two sorted arrays.

### seqMerge()
Sequential recursive merge sort.

### parMerge()
Parallel merge sort using tasks.

---

## Why use depth parameter?

Too many tasks create overhead.

Depth limits recursion parallelization.

---

## Why use #pragma omp single?

Only one thread should create recursive tasks.

---

## Why taskwait important?

Merge should happen only after sorting completes.

---

## Complexity

Time Complexity:
O(n log n)

Space Complexity:
O(n)

---

## Advantages

- Efficient sorting
- Good parallel scalability

---

## Disadvantages

- Extra memory usage
- Task overhead

---

## Important Viva Questions

### Why merge sort faster than bubble sort?
Because complexity is O(n log n).

### Why dynamic arrays used?
Temporary arrays needed during merging.

### What if taskwait removed?
Incorrect output due to incomplete sorting.

---

# 8. PARALLEL BFS USING OPENMP

## Aim

To implement parallel Breadth First Search using OpenMP.

---

## What is BFS?

Breadth First Search traverses graph level by level.

Uses queue data structure.

---

## BFS Characteristics

- FIFO traversal
- Shortest path in unweighted graph
- Level-order traversal

---

## Graph Representation

Adjacency matrix used.

### Advantages
- Simple implementation

### Disadvantages
- High memory usage

---

## Sequential BFS

Steps:
1. Push start node
2. Mark visited
3. Pop node
4. Visit neighbors
5. Repeat

---

## Parallel BFS

Different frontier nodes processed simultaneously.

---

## Why critical section used?

Multiple threads may:
- Update visited array
- Push into queue simultaneously

Critical section prevents race conditions.

---

## Complexity

Adjacency Matrix:
O(V²)

Adjacency List:
O(V + E)

---

## Important Viva Questions

### Why queue used in BFS?
FIFO ensures level-order traversal.

### Can BFS detect shortest path?
Yes in unweighted graphs.

### Why adjacency matrix consumes more memory?
Needs V×V storage.

---

# 9. PARALLEL DFS USING OPENMP

## Aim

To implement parallel Depth First Search using OpenMP tasks.

---

## What is DFS?

DFS explores graph deeply before backtracking.

Uses:
- Stack
OR
- Recursion

---

## DFS Characteristics

- Depth-wise traversal
- Recursive nature
- Backtracking

---

## Parallel DFS

Independent branches explored simultaneously.

---

## Why tasks used?

Different graph branches can execute independently.

---

## Why critical section used?

Prevents multiple threads from visiting same node.

---

## Complexity

Adjacency Matrix:
O(V²)

Adjacency List:
O(V + E)

---

## BFS vs DFS

| BFS | DFS |
|---|---|
| Queue | Stack/Recursion |
| Level traversal | Depth traversal |
| Finds shortest path | Does not guarantee shortest path |
| FIFO | LIFO |

---

## Important Viva Questions

### Why recursion used in DFS?
Recursion naturally implements stack behavior.

### What happens without visited array?
Infinite loops possible.

### Which uses more memory BFS or DFS?
BFS generally uses more memory.

---

# 10. CUDA VECTOR ADDITION

## Aim

To perform vector addition using CUDA.

---

## Theory

Each thread computes:

C[i] = A[i] + B[i]

---

## Why vector addition ideal for GPU?

Operations are independent.

Highly parallel.

---

## CUDA Execution Steps

1. Allocate host memory
2. Allocate device memory
3. Copy data CPU → GPU
4. Launch kernel
5. Copy results GPU → CPU
6. Free memory

---

## Important CUDA Terms

### blockDim.x
Threads per block.

### threadIdx.x
Thread number.

### blockIdx.x
Block number.

---

## Global Index Formula

index = blockIdx.x * blockDim.x + threadIdx.x

---

## Important Viva Questions

### Why synchronization needed?
Ensure kernel execution completes.

### What if thread index exceeds array size?
Out-of-bounds memory access occurs.

### Why GPU faster?
Thousands of threads execute simultaneously.

---

# 11. CUDA MATRIX MULTIPLICATION

## Aim

To perform matrix multiplication using CUDA.

---

## Formula

C[i][j] = Σ A[i][k] × B[k][j]

---

## Complexity

O(n³)

---

## Parallelization

Each thread computes one output element.

---

## Why matrix multiplication important?

Used in:
- Deep learning
- Graphics
- Scientific computing

---

## Optimization

Shared memory improves performance.

---

## Important Viva Questions

### Why matrix multiplication computationally expensive?
Large number of multiply-add operations.

### Why shared memory faster?
Stored on-chip.

### Why GPU ideal?
Massive data parallelism.

---

# 12. CUDA REDUCTION

## Aim

To implement parallel reduction using CUDA.

---

## What is Reduction?

Combining multiple values into one.

Examples:
- Sum
- Max
- Min

---

## Why Reduction Important?

Used in:
- AI
- Statistics
- Scientific computations

---

## Parallel Reduction

Partial sums computed simultaneously.

Then merged.

---

## Shared Memory Usage

Threads in same block share fast memory.

---

## Synchronization

Threads must synchronize after partial computations.

---

## Important Viva Questions

### Why reduction challenging?
Requires synchronization.

### Why shared memory used?
Faster than global memory.

### What is tree reduction?
Pairwise reduction structure.

---

# 13. VERY IMPORTANT THEORY QUESTIONS

## What is Load Balancing?

Equal distribution of work among processors.

---

## What is Deadlock?

Processes wait forever for resources.

---

## What is Starvation?

Thread never gets resources.

---

## What is Cache Memory?

Small fast memory storing frequently used data.

---

## Why locality important?

Better cache performance improves speed.

---

## Shared Memory Architecture

All processors share same memory.

Example:
OpenMP

---

## Distributed Memory Architecture

Each processor has separate memory.

Communication required.

Example:
MPI clusters

---

## What is MPI?

Message Passing Interface.

Used in distributed computing.

---

## OpenMP vs MPI

| OpenMP | MPI |
|---|---|
| Shared memory | Distributed memory |
| Threads | Processes |
| Easier programming | Better scalability |

---

## OpenMP vs CUDA

| OpenMP | CUDA |
|---|---|
| CPU parallelism | GPU parallelism |
| Shared memory multiprocessing | Massive threading |
| Easier | More powerful for data parallelism |

---

# 14. PRACTICAL OUTPUT QUESTIONS

## Why parallel time fluctuates?

Because of:
- CPU scheduling
- Cache effects
- Background processes

---

## Why speedup not linear?

Due to:
- Synchronization overhead
- Sequential portions
- Communication cost

---

## Why parallel program can become slower?

Overhead may exceed computation gain.

---

## Why same random seed used?

For reproducible results.

---

## Why compare sequential and parallel versions?

To evaluate performance improvement.

---

# 15. COMPILATION COMMANDS

## OpenMP Programs

```bash
g++ -fopenmp filename.cpp -o output
./output
```

---

## CUDA Programs

```bash
nvcc filename.cu -o output
./output
```

---

# 16. IMPORTANT DEFINITIONS

## HPC
Using powerful systems for high-speed computation.

---

## Parallel Computing
Executing multiple computations simultaneously.

---

## Thread
Smallest unit of execution.

---

## Kernel
GPU function executed in parallel.

---

## Synchronization
Coordination among threads.

---

## Speedup
Performance improvement ratio.

---

## Efficiency
Processor utilization measure.

---

## Race Condition
Incorrect results due to simultaneous access.

---

## Critical Section
Code executed by one thread at a time.

---

# 17. MOST IMPORTANT LAST-MINUTE QUESTIONS

## Why GPUs are better for HPC?

Because GPUs support thousands of parallel threads.

---

## Why merge sort preferred over bubble sort?

Better complexity:
O(n log n)

---

## Which traversal finds shortest path?

BFS in unweighted graphs.

---

## What is purpose of taskwait?

Ensures child tasks complete before continuing.

---

## Why synchronization needed in parallel programming?

To avoid incorrect results.

---

## Main challenge in parallel computing?

- Synchronization
- Communication
- Load balancing
- Correctness

---

# 18. EXTERNAL EXAMINER RAPID FIRE QUESTIONS

## Define OpenMP.
API for shared-memory parallel programming.

## Define CUDA.
NVIDIA platform for GPU computing.

## Define BFS.
Breadth-first graph traversal.

## Define DFS.
Depth-first graph traversal.

## Define scalability.
Ability to maintain performance with increasing workload.

## Define throughput.
Amount of work completed per unit time.

## Define latency.
Delay before execution starts.

## Define deadlock.
Processes waiting forever for resources.

## Define concurrency.
Multiple tasks making progress together.

## Define parallelism.
Multiple tasks executing simultaneously.

---

# 19. HOW TO EXPLAIN YOUR PRACTICAL IN EXAM

## Bubble Sort
“This program implements sequential and parallel odd-even bubble sort using OpenMP. The array is divided into odd and even phases so that comparisons can happen in parallel safely. Execution times are measured using omp_get_wtime and speedup and efficiency are calculated.”

---

## Merge Sort
“This program implements sequential and parallel merge sort using OpenMP tasks. Recursive halves are processed in parallel and then merged. Taskwait ensures sorting completes before merging.”

---

## BFS
“This program implements sequential and parallel Breadth First Search using OpenMP. Different frontier nodes are explored simultaneously while critical sections prevent race conditions.”

---

## DFS
“This program implements parallel DFS using OpenMP tasks where different graph branches are explored simultaneously.”

---

## Vector Addition
“This CUDA program performs vector addition where each GPU thread computes one output element independently.”

---

## Matrix Multiplication
“This CUDA program performs matrix multiplication where each thread computes one element of output matrix.”

---

## Reduction
“This CUDA program performs parallel reduction where partial sums are computed in parallel and combined efficiently.”

---

# 20. FINAL QUICK REVISION

## OpenMP Keywords
- parallel
- for
- critical
- task
- taskwait
- single

---

## CUDA Keywords
- __global__
- cudaMalloc
- cudaMemcpy
- cudaFree
- threadIdx.x
- blockIdx.x
- blockDim.x

---

## Important Complexities

### Bubble Sort
O(n²)

### Merge Sort
O(n log n)

### BFS/DFS using adjacency matrix
O(V²)

### Matrix Multiplication
O(n³)

---

# END OF GUIDE

