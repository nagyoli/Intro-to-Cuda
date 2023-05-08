
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>

#define N 2000

float A[N];
float B[N];
float C[N];
float temp[N];
__device__ float dev_B[N];
__shared__ float shr_B[N];
__shared__ float shr_temp[N];

void averageCPU() {

    temp[0] = (0 + A[1]) / 2.0;

    for (size_t i = 1; i < N - 1; i++)
    {
        temp[i] = (A[i - 1] + A[i + 1]) / 2.0;
    }

    temp[N - 1] = (A[N - 2] + 0) / 2.0;

    for (size_t i = 0; i < N; i++)
    {
        C[i] = temp[i];
    }
}

__global__ void averageOfNeighbours() {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    shr_B[i] = dev_B[i];


    if (i == 0)
        shr_temp[i] = (0 + shr_B[i + 1]) / 2.0;


    if (i == N - 1)
        shr_temp[i] = (shr_B[i - 1] + 0) / 2.0;

    if (i != 0 && i != N - 1)
        shr_temp[i] = (shr_B[i - 1] + shr_B[i + 1]) / 2.0;


    dev_B[i] = shr_temp[i];

}

int main()
{
    // Tömbök feltöltése véletlen számokkal
    const int range_from = 0;
    const int range_to = 100;
    std::random_device                  rand_dev;
    std::mt19937                        generator(rand_dev());
    std::uniform_int_distribution<int>  distr(range_from, range_to);
    int elem;

    for (size_t i = 0; i < N; i++)
    {
        elem = distr(generator);
        B[i] = elem;
        A[i] = elem;

    }

    averageCPU();

    int threadsInBlock = 1000;
    int blockNumber = (N - 1) / threadsInBlock + 1;

    cudaMemcpyToSymbol(dev_B, B, N * sizeof(float));

    averageOfNeighbours << <blockNumber, threadsInBlock >> > ();

    cudaMemcpyFromSymbol(B, dev_B, N * sizeof(float));


    // GPU-n átlagolt tömb
    printf("\n");
    for (size_t i = 0; i < N; i++)
    {
        printf("A[%d]: %.2f\t C[%d]: %.2f\t B[%d]: %.2f\n", i, A[i], i, C[i], i, B[i]);
    }

    return 0;
}
