
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


#include <iostream>
#include <ctime>
#include <cstdlib>

const int N = 8;
const int BLOCK_SIZE = N;
const int TILE_WIDTH = 4;


__global__ void matrixMulKernel(int* A, int* B, int* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0;
    if (row < N && col < N) {
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

__global__ void tiledMatrixMulKernel(int* A, int* B, int* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float shared_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_B[TILE_WIDTH][TILE_WIDTH];

    float value = 0;

    for (int tileIdx = 0; tileIdx < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++tileIdx) {
        if (row < N && tileIdx * TILE_WIDTH + threadIdx.x < N) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * N + tileIdx * TILE_WIDTH + threadIdx.x];
        }
        else {
            shared_A[threadIdx.y][threadIdx.x] = 0;
        }

        if (col < N && tileIdx * TILE_WIDTH + threadIdx.y < N) {
            shared_B[threadIdx.y][threadIdx.x] = B[(tileIdx * TILE_WIDTH + threadIdx.y) * N + col];
        }
        else {
            shared_B[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            value += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}


void multiplyMatrices(const int* A, const int* B, int* C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float value = 0;
            for (int k = 0; k < N; ++k) {
                value += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = value;
        }
    }
}

void generateRandomMatrix(int* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = rand() % 100;
    }
}

void print_matrix(const int* matrix, int N) {
    

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << matrix[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    srand(time(0));

    int* h_A, * h_B, * h_C, *h_D, *h_E;
    int* d_A, * d_B, * d_D, *d_E;
    size_t bytes = N * N * sizeof(int);

    h_A = (int*)malloc(bytes);
    h_B = (int*)malloc(bytes);
    h_C = (int*)malloc(bytes);
    h_D = (int*)malloc(bytes);
    h_E = (int*)malloc(bytes);

    generateRandomMatrix(h_A, N * N);
    generateRandomMatrix(h_B, N * N);
    multiplyMatrices(h_A, h_B, h_C);

    std::cout << "1st matrix" << std::endl;
    print_matrix(h_A, N);
    std::cout << std::endl;

    std::cout << "2st matrix" << std::endl;
    print_matrix(h_B, N);
    std::cout << std::endl;

    std::cout << "CPU multiplied matrix" << std::endl;
    print_matrix(h_C, N);
    std::cout << std::endl;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_D, bytes);
    cudaMalloc(&d_E, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);


    dim3((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrixMulKernel << < dim3((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (d_A, d_B, d_D);

    cudaMemcpy(h_D, d_D, bytes, cudaMemcpyDeviceToHost);

    std::cout << "GPU multiplied matrix" << std::endl;
    print_matrix(h_D, N);
    std::cout << std::endl;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    tiledMatrixMulKernel << <gridDim, blockDim >> > (d_A, d_B, d_E);

    cudaMemcpy(h_E, d_E, bytes, cudaMemcpyDeviceToHost);

    std::cout << "GPU tiled-multipled matrix" << std::endl;
    print_matrix(h_E, N);
    std::cout << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D);
    cudaFree(d_E);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    free(h_E);

    return 0;
}