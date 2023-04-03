
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define N 20

#define M 3

char text[N + 1] = "Ez egy hosszu szoveg";
char word[M + 1] = "szo";
int pos = -1;

__device__ char dev_text[N];
__device__ char dev_word[M];
__device__ char dev_temp[N];
__device__ int dev_pos;

void findWordCPU() {
	pos = -1;
	for (size_t i = 0; i < N - M; i++)
	{
		int j = 0;

		while (j < M && text[i + j] == word[j])
		{
			j++;
		}
		if (j == M)
		{
			pos = i;
		}
	}
	printf("CPU result: %d\n", pos);
}

__global__ void findWordSingleThread() {

	dev_pos = -1;
	for (int i = 0; i < N - M; i++)
	{
		int j = 0;

		while (j < M && dev_text[i + j] == dev_word[j])
		{
			j++;
		}
		if (j == M)
		{
			dev_pos = i;
		}
	}

}

__global__ void findWordN() {

	dev_pos = -1;
	int i = threadIdx.x;
	int j = 0;

	while (j < M && dev_text[i + j] == dev_word[j])
	{
		j++;
	}
	if (j == M)
	{
		dev_pos = i;
	}

}


__global__ void findWordNM() {

	if (threadIdx.y == 0)
	{
		dev_temp[threadIdx.x] = 0;
	}

	__syncthreads();

	if (dev_text[threadIdx.x + threadIdx.y] != dev_word[threadIdx.y])
	{
		dev_temp[threadIdx.x] = 1;
	}

	__syncthreads();

	if (threadIdx.y == 0)
	{
		if (dev_temp[threadIdx.x] == 0)
		{
			dev_pos = threadIdx.x;
		}
	}

	__syncthreads();

}

__global__ void findWordNMShared(){
	__shared__ char shr_text[N];
	__shared__ char shr_word[M];
	__shared__ char shr_temp[N];


	int i = threadIdx.x + threadIdx.y * blockDim.x;

	if (i < N)
		shr_text[i] = dev_temp[i];


	if (threadIdx.x == 0)
		shr_word[threadIdx.y] = dev_word[threadIdx.y];

	if (threadIdx.y == 0)
		shr_temp[threadIdx.x] = 0;
	
	__syncthreads();

	if (shr_text[threadIdx.x + threadIdx.y] != shr_word[threadIdx.y])
		shr_temp[threadIdx.x] = 1;

	__syncthreads();

	if (threadIdx.y == 0)
		if (shr_temp[threadIdx.x] == 0)
			dev_pos = threadIdx.x;

	__syncthreads();
}


int main()
{

	findWordCPU();

	cudaMemcpyToSymbol(dev_text, text, N * sizeof(char));
	cudaMemcpyToSymbol(dev_word, word, M * sizeof(char));

	findWordSingleThread << <1, 1 >> > ();
	cudaMemcpyFromSymbol(&pos, dev_pos, sizeof(int));
	printf("GPU single thread result: %d\n", pos);

	findWordN << <1, N - M + 1 >> > ();
	cudaMemcpyFromSymbol(&pos, dev_pos, sizeof(int));
	printf("GPU N-M thread result: %d\n", pos);


	findWordNM << <1, dim3(N - M + 1, M) >> > ();
	cudaMemcpyFromSymbol(&pos, dev_pos, sizeof(int));
	printf("GPU NxM thread result: %d\n", pos);


	findWordNMShared << <1, dim3(N - M + 1, M) >> > ();
	cudaMemcpyFromSymbol(&pos, dev_pos, sizeof(int));
	printf("GPU NxM thread Shared result: %d\n", pos);


	return 0;
}
