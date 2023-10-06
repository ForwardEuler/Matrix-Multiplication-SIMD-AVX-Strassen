#include <vector_types.h>

/**
* @brief Implements an NxN matrix multiply C=A*B
*
* @param[in] N : dimension of square matrix (NxN)
* @param[in] A : pointer to input NxN matrix
* @param[in] B : pointer to input NxN matrix
* @param[out] C : pointer to output NxN matrix
* @param[in] flags : pointer to array of integers which can be used for debugging and performance tweaks. Optional. If unused, set to zero
* @param[in] flagCount : the length of the flags array
* @return void
* */
__host__ void matrixMultiply_GPU(int N, const float* A, const float* B, float* C, int* flags, int flagCount);

__global__ void matrixMultiplyKernel_GPU(int N, const float* A, const float* B, float* C, int flag0, int flag1, int flag2);