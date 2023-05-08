#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

const int N = 23;
const int K = 4;

__device__ int devSzamok[N];
__device__ int devResult[N - K + 1];
__device__ int devIndex;
__device__ int devMin;

__global__ void OsszegezesNBlockKKSzallal()
{
	__shared__ int shr_Szamok[2 * K];

	if (threadIdx.y == 0)
	{
		shr_Szamok[threadIdx.x] = devSzamok[blockIdx.x * K + threadIdx.x];
		shr_Szamok[threadIdx.x + K] = devSzamok[blockIdx.x * K + threadIdx.x + K];

		devResult[blockIdx.x * K + threadIdx.x] = 0;
	}

	__syncthreads();

	atomicAdd(&(devResult[blockIdx.x * K + threadIdx.x]), shr_Szamok[threadIdx.x + threadIdx.y]);
}

__global__ void MinKivalasztasNSzallal()
{
	if (threadIdx.x == 0)
	{
		devMin = devResult[0];
		devIndex = 0;
	}

	if (atomicMin(&devMin, devResult[threadIdx.x]) != devMin)
	{
		devIndex = threadIdx.x;
	}
}

int main()
{
	//cudaFuncSetCacheConfig(,cudaFuncCachePreferShared);

	int szamok[] = { 3, 7, 1, 4, 7, 10, 2, 9, 16, 3, 3, 5, 7, 9, 1, 0, 1, 7, 8, 7, 6, 5, 1 };
	int szamokSize = sizeof szamok / sizeof szamok[0];
	int result[N - K + 1];
	int index;
	int min;

	cudaMemcpyToSymbol(devSzamok, szamok, sizeof(int) * szamokSize);

	//Amilyen hosszú az összegzendõ szakasz (K), annyival osztom a teljes számsor méretét, annyi blokkot indítok. Ehhez mérten egy blokkon belül K*K szál indul.

	OsszegezesNBlockKKSzallal << < ((N - 1) / K) + 1, dim3(K, K) >> > ();

	MinKivalasztasNSzallal << < 1, N - K >> > ();

	cudaMemcpyFromSymbol(result, devResult, sizeof(int) * (N - K + 1));
	cudaMemcpyFromSymbol(&min, devMin, sizeof(int));
	cudaMemcpyFromSymbol(&index, devIndex, sizeof(int));

	printf("Minimum ezen az indexen: %d\n", index);
	printf("Minimum ertek: %d\n", min);

	//CUDA Occupancy Calculatorban 16 szál/blokkal számoltam, 0 register használattal, és 32 byte-nyi shared memóriával, (8 (2*4) int blokkonként). Ez alapján 33%-os kihasználtságot ír blokkonként.

	return 0;
}

