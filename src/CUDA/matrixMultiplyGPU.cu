/**
 * Author: Chuan Tian
 * Date: Sep 28, 2023
 * Student ID: 45108907
 */
#include "matrixMultiplyGPU.cuh"

typedef char int8;
typedef short int16;
typedef int int32;
typedef long long int64;

typedef unsigned char byte;
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

//uncomment the line below to run stand-alone test on local machine
//#define LOCALTEST
#ifndef LOCALTEST
#define FORTRAN_ORDER
#endif

constexpr int TILE_WIDTH = 32;

__global__ void matrixMultiplyKernel_GPU(int N, const float* A, const float* B, float* C)
{
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    const u32 block_x = blockIdx.x, block_y = blockIdx.y;
    const u32 thread_x = threadIdx.x, thread_y = threadIdx.y;

    const u32 col_x = block_x * TILE_WIDTH + thread_x;
    const u32 row_y = block_y * TILE_WIDTH + thread_y;

    float sum = 0;
    const u32 numTiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int tileId = 0; tileId < numTiles; tileId++)
    {
        // copy
        if (row_y < N && tileId * TILE_WIDTH + thread_x < N)
            s_A[thread_y][thread_x] = A[row_y * N + tileId * TILE_WIDTH + thread_x];
        else
            s_A[thread_y][thread_x] = 0;

        if (col_x < N && tileId * TILE_WIDTH + thread_y < N)
            s_B[thread_y][thread_x] = B[(tileId * TILE_WIDTH + thread_y) * N + col_x];
        else
            s_B[thread_y][thread_x] = 0;

        __syncthreads();

        // Compute for one tile
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += s_A[thread_y][k] * s_B[k][thread_x];
        }
        __syncthreads();
    }

    if (row_y < N && col_x < N) {
        C[row_y * N + col_x] = sum;
    }
}

__host__ void matrixMultiply_GPU(int N, const float* A, const float* B, float* C, int* flags, int flagCount)
{
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

#ifndef FORTRAN_ORDER
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);
#else
    cudaMemcpy(d_A, B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
#endif

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);  // This can be tuned for performance
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMultiplyKernel_GPU<<<blocksPerGrid, threadsPerBlock>>>(N, d_A, d_B, d_C);

    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

#ifdef LOCALTEST

#include <iostream>
#include <chrono>
#include <cmath>
#include <random>

// CPU matrix multiplication
__host__ void matrixMultiply_CPU(int N, const float* A, const float* B, float* C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main()
{
    const int N = 1024;
    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C_CPU = new float[N * N];
    float* C_GPU = new float[N * N];
    float* tmp = new float[N * N];

    std::random_device rd;
    std::mt19937 gen(rd());

    // Create a uniform real distribution between -10 and 10
    std::uniform_real_distribution<> dis(-10.0, 10.0);

    for (int i = 0; i < N * N; i++) {
        A[i] = (float)dis(gen);
        B[i] = (float)dis(gen);
    }

    // Measure CPU time
    auto startCPU = std::chrono::high_resolution_clock::now();
    matrixMultiply_CPU(N, A, B, C_CPU);
    auto endCPU = std::chrono::high_resolution_clock::now();
    auto durationCPU = std::chrono::duration_cast<std::chrono::milliseconds>(endCPU - startCPU).count();

    // Measure GPU time
    matrixMultiply_GPU(N, A, B, tmp, nullptr, 0);
    auto startGPU = std::chrono::high_resolution_clock::now();
    matrixMultiply_GPU(N, A, B, C_GPU, nullptr, 0);
    auto endGPU = std::chrono::high_resolution_clock::now();
    auto durationGPU = std::chrono::duration_cast<std::chrono::milliseconds>(endGPU - startGPU).count();

    // Validate GPU result using RMSE
    double sse = 0.0;
    for (int i = 0; i < N * N; i++) {
        double error = C_CPU[i] - C_GPU[i];
        sse += error * error;
    }
    double rmse = sqrt(sse / (N * N));

    bool valid = rmse < 1e-2;

    std::cout << "CPU Time: " << durationCPU << " ms\n";
    std::cout << "GPU Time: " << durationGPU << " ms\n";
    std::cout << "RMSE: " << rmse << "\n";
    std::cout << "GPU result is " << (valid ? "valid" : "invalid") << "\n";
    
    delete[] A;
    delete[] B;
    delete[] C_CPU;
    delete[] C_GPU;

    return 0;
}
#endif