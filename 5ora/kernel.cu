
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <time.h>

const int N = 23;

__device__ int dev_input[N];
__device__ int dev_min_val;

__global__ void findMinGPUMParallelReduction() {
    dev_min_val = INT_MAX;
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    for (int csusztatas = 1; csusztatas < blockDim.x; csusztatas *= 2) {
        if (local_tid % (2 * csusztatas) == 0 && globalIndex + csusztatas < N) {
            dev_input[globalIndex] = min(dev_input[globalIndex], dev_input[globalIndex + csusztatas]);
        }
        __syncthreads();
    }

    if (local_tid == 0) {
        atomicMin(&dev_min_val, dev_input[globalIndex]);
    }
}

__global__ void findMinGPUMultiThreadAtomicSharedMemory() {
    dev_min_val = INT_MAX;
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    // Allocate shared memory for the block
    __shared__ int shared_min_val;

    if (local_tid == 0) {
        shared_min_val = dev_input[globalIndex];
    }

    __syncthreads();

    if (globalIndex < N) {
        atomicMin(&shared_min_val, dev_input[globalIndex]);
    }

    __syncthreads();

    if (local_tid == 0) {
        atomicMin(&dev_min_val, shared_min_val);
    }
}

__global__ void findMinGPUMultiThreadAtomic() {
    dev_min_val = INT_MAX;
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    atomicMin(&dev_min_val, dev_input[globalIndex]);
}

__global__ void findMinGPUSingleThread() {
    dev_min_val = dev_input[0];

    for (int i = 1; i < N; i++) {
        if (dev_input[i] < dev_min_val) {
            dev_min_val = dev_input[i];
        }
    }

}

int findMinCPU(int input[N]) {
    int min_val = input[0];
    for (int i = 1; i < N; i++) {
        if (input[i] < min_val) {
            min_val = input[i];
        }
    }

    return min_val;
}

int main()
{
    clock_t start, end;
    double cpu_time_taken, gpu1_time_taken, gpu2_time_taken, gpu3_time_taken, gpu4_time_taken, gpu5_time_taken;

    int input[] = {3, 7, 1, 4, 7, 10, 2, 9, 16, 3, 3, 5, 7, 9, 1, 0, 1, 7, 8, 7, 6, 5, 1};
    int min_val = INT_MAX;


    // CPU
    start = clock();
    min_val = findMinCPU(input);
    end = clock();
    cpu_time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Legkisebb ertek CPU: %d\n", min_val);

    // Single Thread
    cudaMemcpyToSymbol(dev_input, input, N * sizeof(int));
    start = clock();
    findMinGPUSingleThread << <1, 1 >> > ();
    end = clock();
    cudaMemcpyFromSymbol(&min_val, dev_min_val, sizeof(int));
    gpu1_time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Legkisebb ertek GPU 1thred: %d\n\n", min_val);


    int numberOfThreads = 4;
    int numberOfBlocks = (N + numberOfThreads - 1) / numberOfThreads;
    
    // Multi-thread Atomic
    start = clock();
    findMinGPUMultiThreadAtomic << <numberOfBlocks, numberOfThreads >> > ();
    end = clock();
    cudaMemcpyFromSymbol(&min_val, dev_min_val, sizeof(int));
    gpu2_time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Legkisebb ertek GPU atomic: %d\n", min_val);


    // Multi-thread shared memory
    start = clock();
    findMinGPUMultiThreadAtomicSharedMemory << <numberOfBlocks, numberOfThreads >> > ();
    end = clock();
    cudaMemcpyFromSymbol(&min_val, dev_min_val, sizeof(int));
    gpu3_time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Legkisebb ertek GPU shared: %d\n", min_val);


    // Parallel Reduction
    start = clock();
    findMinGPUMParallelReduction << <numberOfBlocks, numberOfThreads >> > ();
    end = clock();
    cudaMemcpyFromSymbol(&min_val, dev_min_val, sizeof(int));
    gpu4_time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Legkisebb ertek GPU pr: %d\n", min_val);

    
    return 0;
}
