#include <thread> //used for getting the number of cpu cores available (this is the hardware core count, not the slurm count)
#include <omp.h> //openMP library
#include <immintrin.h>//AVX library

#include <string.h> //This is included mostly so you can memset, e.g. for setting arrays to zero

//You won't have access to these libraries in the final version, but for debugging you might want to use them temporarily
//#include <stdio.h>
//#include <iostream>

/**
* @brief Implements an NxN matrix multiply C=A*B
*
* @param[in] N : dimension of square matrix (NxN)
* @param[in] A : pointer to input NxN matrix
* @param[in] B : pointer to input NxN matrix
* @param[out] C : pointer to output NxN matrix
* @param[in] args : pointer to array of integers which can be used for debugging and performance tweaks. Optional. If unused, set to zero
* @param[in] argCount : the length of the flags array
* @return void
* */
void matrixMultiply(int N, const float* A, const float* B, float* C, int* args, int argCount);