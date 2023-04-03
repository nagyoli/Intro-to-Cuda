
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

int A[] = { 1, 2, 3, 4, 5 };
int B[] = { 2, 4, 6, 8, 10 };
int C[] = { 0, 0, 0, 0, 0 };

__device__ int dev_A[5];
__device__ int dev_B[5];
__device__ int dev_C[5];

__global__ void szorzas(int mennyivel) {
	int i = threadIdx.x;
	dev_A[i] *= mennyivel;
}

__global__ void vector_elemwise_add() {
	int i = threadIdx.x;
	dev_C[i] = dev_A[i] + dev_B[i];
}


int main()
{
    cudaMemcpyToSymbol(dev_A, A, 5 * sizeof(int));
	cudaMemcpyToSymbol(dev_B, B, 5 * sizeof(int));
	cudaMemcpyToSymbol(dev_C, C, 5 * sizeof(int));

	szorzas <<<1, 5 >>> (3);
	vector_elemwise_add <<<1, 5 >> > ();

	cudaMemcpyFromSymbol(A, dev_A, 5 * sizeof(int));
	cudaMemcpyFromSymbol(C, dev_C, 5 * sizeof(int));



	for (size_t i = 0; i < 5; i++)
	{
		printf("A[%d]=%d\n", i, A[i]);
	}

	printf("\n");

	for (size_t i = 0; i < 5; i++)
	{
		printf("C[%d]=%d\n", i, C[i]);
	}

    return 0;
}
