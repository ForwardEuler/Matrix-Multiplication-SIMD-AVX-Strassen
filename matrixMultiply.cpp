#include "matrixMultiply.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <chrono>
#include <cassert>
#include <cstdlib>
#include <immintrin.h>
#include <thread>
#include "mem_pool.h"


//void matrixMultiply(int N, const float* A, const float* B, float* C, int* args, int argCount);
void matrixPadding(int N, const float* A, const float* B, float* C, int* args, int argCount);

constexpr int THRESHOLD = 64;

inline void add(const float* A, const float* B, float* C, int N) {
    const int l = N * N;
    for (int i = 0; i < l; i++) {
        C[i] = A[i] + B[i];
    }
}

inline void subtract(const float* A, const float* B, float* C, int N) {
    const int l = N * N;
    for (int i = 0; i < l; i++) {
        C[i] = A[i] - B[i];
    }
}

inline void add_AVX(const float* A, const float* B, float* C, int N) {
    const int l = N * N;
    if (l <= THRESHOLD)
    {
        add(A, B, C, N);
        return;
    }
    int i;
    // add eight floats at a time
    for (i = 0; i <= l - 8; i += 8) {
        __m256 v8f_A = _mm256_loadu_ps(&A[i]);
        __m256 v8f_B = _mm256_loadu_ps(&B[i]);
        __m256 v8f_C = _mm256_add_ps(v8f_A, v8f_B);
        _mm256_storeu_ps(&C[i], v8f_C);
    }

    // add remaining floats
    for (; i < l; i++) {
        C[i] = A[i] + B[i];
    }
}

inline void subtract_AVX(const float* A, const float* B, float* C, int N) {
    const int l = N * N;
    if (l <= THRESHOLD)
    {
        subtract(A, B, C, N);
        return;
    }
    int i;
    // subtract eight floats at a time
    for (i = 0; i <= l - 8; i += 8) {
        __m256 v8f_A = _mm256_loadu_ps(&A[i]);
        __m256 v8f_B = _mm256_loadu_ps(&B[i]);
        __m256 v8f_C = _mm256_sub_ps(v8f_A, v8f_B);
        _mm256_storeu_ps(&C[i], v8f_C);
    }

    // subtract remaining floats
    for (; i < l; i++) {
        C[i] = A[i] - B[i];
    }
}

inline void strassenMultiply(const float* A, const float* B, float* C)
{
    // ref: https://en.wikipedia.org/wiki/Strassen_algorithm
    float M1, M2, M3, M4, M5, M6, M7;
    // M1 = (A11 + A22) * (B11 + B22)
    M1 = (A[0] + A[3]) * (B[0] + B[3]);
    // M2 = (A21 + A22) * B11
    M2 = (A[2] + A[3]) * B[0];
    // M3 = A11 * (B12 - B22)
    M3 = A[0] * (B[1] - B[3]);
    // M4 = A22 * (B21 - B11)
    M4 = A[3] * (B[2] - B[0]);
    // M5 = (A11 + A12) * B22
    M5 = (A[0] + A[1]) * B[3];
    // M6 = (A21 - A11) * (B11 + B12)
    M6 = (A[2] - A[0]) * (B[0] + B[1]);
    // M7 = (A12 - A22) * (B21 + B22)
    M7 = (A[1] - A[3]) * (B[2] + B[3]);

    C[0] = M1 + M4 - M5 + M7;
    C[1] = M3 + M5;
    C[2] = M2 + M4;
    C[3] = M1 - M2 + M3 + M6;
}

void thread_work(int N, const float* A, const float* B_transpose, float* C, int i, int end)
{
    for (; i < end; i++) {
        for (int j = 0; j < N; j++) {
            __m256 sum = _mm256_setzero_ps(); // Initialize sum to zero
            int k;
            for (k = 0; k <= N - 8; k += 8) {
                // Load 8 floats from A and B_transpose
                __m256 a = _mm256_loadu_ps(&A[i * N + k]);
                __m256 b = _mm256_loadu_ps(&B_transpose[j * N + k]);

                // Multiply the vectors element-wise
                __m256 prod = _mm256_mul_ps(a, b);

                // Add to the sum
                sum = _mm256_add_ps(sum, prod);
            }

            // Horizontal add to get the sum of the eight floats in the vector
            sum = _mm256_hadd_ps(sum, sum);
            sum = _mm256_hadd_ps(sum, sum);

            float result[8];
            _mm256_storeu_ps(result, sum);

            C[i * N + j] = result[0] + result[4];

            // Handle the case when N is not a multiple of 8
            for (; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B_transpose[j * N + k];
            }
        }
    }
}


void msize_mul(int N, const float* A, const float* B, float* C) {
    // Transpose B into B'
    auto* B_transpose = new float[N * N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B_transpose[j * N + i] = B[i * N + j];
        }
    }

    int block_size = N / 4;
    int remainder = N % 4;
    std::thread t1(thread_work, N, A, B_transpose, C, 0, block_size);
    std::thread t2(thread_work, N, A, B_transpose, C, block_size, block_size * 2);
    std::thread t3(thread_work, N, A, B_transpose, C, block_size * 2, block_size * 3);
    std::thread t4(thread_work, N, A, B_transpose, C, block_size * 3, block_size * 4 + remainder);

    t1.join();
    t2.join();
    t3.join();
    t4.join();
    delete[] B_transpose;
}

inline void fast_mul(int N, const float* A, const float* B, float* C) {
    float* B_transpose = new float[N * N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B_transpose[j * N + i] = B[i * N + j];
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            __m256 sum = _mm256_setzero_ps(); // Initialize sum to zero
            int k;
            for (k = 0; k <= N - 8; k += 8) {
                // Load 8 floats from A and B_transpose
                __m256 a = _mm256_loadu_ps(&A[i * N + k]);
                __m256 b = _mm256_loadu_ps(&B_transpose[j * N + k]);

                // Multiply the vectors element-wise
                __m256 prod = _mm256_mul_ps(a, b);

                // Add to the sum
                sum = _mm256_add_ps(sum, prod);
            }

            // Horizontal add to get the sum of the eight floats in the vector
            sum = _mm256_hadd_ps(sum, sum);
            sum = _mm256_hadd_ps(sum, sum);

            float result[8];
            _mm256_storeu_ps(result, sum);

            C[i * N + j] = result[0] + result[4];

            // Handle the case when N is not a multiple of 8
            for (; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B_transpose[j * N + k];
            }
        }
    }
}


void matMultiply(int N, const float* A, const float* B, float* C, int* args, int argCount)
{
    // Base case
    if (N <= 2) {
        if (N == 1) {
            C[0] = A[0] * B[0];
            return;
        }
        strassenMultiply(A, B, C);
        return;
    }
    // at N<=256, naive matrix multiplication with avx is faster than Strassen's algorithm
    if (N <= 256) {
        fast_mul(N, A, B, C);
        return;
    }
    const int half_N = N / 2;

    if ((N & (N - 1)) != 0) {
        matrixPadding(N, A, B, C, args, argCount);
        return;
    }

    float *a11, *a12, *a21, *a22;
    float *b11, *b12, *b21, *b22;
    memPool pool1(sizeof(float) * half_N * half_N * 8 + 4);
    u64 alloc_size = sizeof(float) * half_N * half_N;
    a11 = (float *)pool1.memloc(alloc_size);
    a12 = (float *)pool1.memloc(alloc_size);
    a21 = (float *)pool1.memloc(alloc_size);
    a22 = (float *)pool1.memloc(alloc_size);

    b11 = (float *)pool1.memloc(alloc_size);
    b12 = (float *)pool1.memloc(alloc_size);
    b21 = (float *)pool1.memloc(alloc_size);
    b22 = (float *)pool1.memloc(alloc_size);

    for (int i = 0; i < half_N; i++) {
        for (int j = 0; j < half_N; j++) {
            a11[i * half_N + j] = A[i * N + j];
            a12[i * half_N + j] = A[i * N + j + half_N];
            a21[i * half_N + j] = A[(i + half_N) * N + j];
            a22[i * half_N + j] = A[(i + half_N) * N + j + half_N];

            b11[i * half_N + j] = B[i * N + j];
            b12[i * half_N + j] = B[i * N + j + half_N];
            b21[i * half_N + j] = B[(i + half_N) * N + j];
            b22[i * half_N + j] = B[(i + half_N) * N + j + half_N];
        }
    }

    memPool pool2(sizeof(float) * half_N * half_N * 9 + 4);
    float *m1, *m2, *m3, *m4, *m5, *m6, *m7;
    m1 = (float *)pool2.memloc(alloc_size);
    m2 = (float *)pool2.memloc(alloc_size);
    m3 = (float *)pool2.memloc(alloc_size);
    m4 = (float *)pool2.memloc(alloc_size);
    m5 = (float *)pool2.memloc(alloc_size);
    m6 = (float *)pool2.memloc(alloc_size);
    m7 = (float *)pool2.memloc(alloc_size);

    float *r1, *r2;
    r1 = (float *)pool2.memloc(alloc_size);
    r2 = (float *)pool2.memloc(alloc_size);

    // M1 = (A11 + A22) * (B11 + B22)
    add_AVX(a11, a22, r1, half_N);
    add_AVX(b11, b22, r2, half_N);
    matMultiply(half_N, r1, r2, m1, args, argCount);

    // M2 = (A21 + A22) * B11
    add_AVX(a21, a22, r1, half_N);
    matMultiply(half_N, r1, b11, m2, args, argCount);

    // M3 = A11 * (B12 - B22)
    subtract_AVX(b12, b22, r2, half_N);
    matMultiply(half_N, a11, r2, m3, args, argCount);

    // M4 = A22 * (B21 - B11)
    subtract_AVX(b21, b11, r2, half_N);
    matMultiply(half_N, a22, r2, m4, args, argCount);

    // M5 = (A11 + A12) * B22
    add_AVX(a11, a12, r1, half_N);
    matMultiply(half_N, r1, b22, m5, args, argCount);

    // M6 = (A21 - A11) * (B11 + B12)
    subtract_AVX(a21, a11, r1, half_N);
    add_AVX(b11, b12, r2, half_N);
    matMultiply(half_N, r1, r2, m6, args, argCount);

    // M7 = (A12 - A22) * (B21 + B22)
    subtract_AVX(a12, a22, r1, half_N);
    add_AVX(b21, b22, r2, half_N);
    matMultiply(half_N, r1, r2, m7, args, argCount);

    float *temp1, *temp2;
    temp1 = (float *)pool2.memloc(alloc_size);
    temp2 = (float *)pool2.memloc(alloc_size);

    // C11 = M1 + M4 - M5 + M7
    add_AVX(m1, m4, temp1, half_N);
    subtract_AVX(temp1, m5, temp2, half_N);
    add_AVX(temp2, m7, temp1, half_N);
    for (int i = 0; i < half_N; i++)
        for (int j = 0; j < half_N; j++)
            C[i * N + j] = temp1[i * half_N + j];

    // C12 = M3 + M5
    add_AVX(m3, m5, temp1, half_N);
    for (int i = 0; i < half_N; i++)
        for (int j = 0; j < half_N; j++)
            C[i * N + j + half_N] = temp1[i * half_N + j];

    // C21 = M2 + M4
    add_AVX(m2, m4, temp1, half_N);
    for (int i = 0; i < half_N; i++)
        for (int j = 0; j < half_N; j++)
            C[(i + half_N) * N + j] = temp1[i * half_N + j];

    // C22 = M1 - M2 + M3 + M6
    subtract_AVX(m1, m2, temp1, half_N);
    add_AVX(temp1, m3, temp2, half_N);
    add_AVX(temp2, m6, temp1, half_N);
    for (int i = 0; i < half_N; i++)
        for (int j = 0; j < half_N; j++)
            C[(i + half_N) * N + j + half_N] = temp1[i * half_N + j];

    pool1.memfree();
    pool2.memfree();
}


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
void matrixMultiply(int N, const float* A, const float* B, float* C, int* args, int argCount) {

//#define SWAP_AB
#ifdef SWAP_AB
    const float* t = B;
    B = A;
    A = t;
#endif

    const int half_N = N / 2;
    // Base case
    if (N <= 2) {
        if (N == 1) {
            C[0] = A[0] * B[0];
            return;
        }
        strassenMultiply(A, B, C);
        return;
    }
    // if N is not a power of 2, call paddedMatrixMultiply
    if ((N & (N - 1)) != 0) {
        matrixPadding(N, A, B, C, args, argCount);
        return;
    }

    float *a11, *a12, *a21, *a22;
    float *b11, *b12, *b21, *b22;

    a11 = new float[half_N * half_N];
    a12 = new float[half_N * half_N];
    a21 = new float[half_N * half_N];
    a22 = new float[half_N * half_N];

    b11 = new float[half_N * half_N];
    b12 = new float[half_N * half_N];
    b21 = new float[half_N * half_N];
    b22 = new float[half_N * half_N];

    std::thread t1([&]() {
        for (int i = 0; i < half_N; i++) {
            for (int j = 0; j < half_N; j++) {
                a11[i * half_N + j] = A[i * N + j];
                b11[i * half_N + j] = B[i * N + j];
            }
        }
    });

    std::thread t2([&]() {
        for (int i = 0; i < half_N; i++) {
            for (int j = 0; j < half_N; j++) {
                a12[i * half_N + j] = A[i * N + j + half_N];
                b12[i * half_N + j] = B[i * N + j + half_N];
            }
        }
    });

    std::thread t3([&]() {
        for (int i = 0; i < half_N; i++) {
            for (int j = 0; j < half_N; j++) {
                a21[i * half_N + j] = A[(i + half_N) * N + j];
                b21[i * half_N + j] = B[(i + half_N) * N + j];
            }
        }
    });

    std::thread t4([&]() {
        for (int i = 0; i < half_N; i++) {
            for (int j = 0; j < half_N; j++) {
                a22[i * half_N + j] = A[(i + half_N) * N + j + half_N];
                b22[i * half_N + j] = B[(i + half_N) * N + j + half_N];
            }
        }
    });

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    //memPool pool2(sizeof(float) * half_N * half_N * 11);
    std::vector<float*> gc;
    float *m1, *m2, *m3, *m4, *m5, *m6, *m7;
    m1 = new float[half_N * half_N];
    m2 = new float[half_N * half_N];
    m3 = new float[half_N * half_N];
    m4 = new float[half_N * half_N];
    m5 = new float[half_N * half_N];
    m6 = new float[half_N * half_N];
    m7 = new float[half_N * half_N];
    gc.insert(gc.end(), {m1, m2, m3, m4, m5, m6, m7});

    float *r11, *r12, *r21, *r31, *r32, *r42;
    float *r51, *r61, *r62, *r71, *r72;

    r11 = new float[half_N * half_N];
    r12 = new float[half_N * half_N];
    r21 = new float[half_N * half_N];
    r32 = new float[half_N * half_N];
    r42 = new float[half_N * half_N];
    r51 = new float[half_N * half_N];
    r61 = new float[half_N * half_N];
    r62 = new float[half_N * half_N];
    r71 = new float[half_N * half_N];
    r72 = new float[half_N * half_N];
    gc.insert(gc.end(), {r11, r12, r21, r32, r42, r51, r61, r62, r71, r72});

    std::thread tm1([&]() {
        add_AVX(a11, a22, r11, half_N);
        add_AVX(b11, b22, r12, half_N);
        matMultiply(half_N, r11, r12, m1, args, argCount);
    });

    std::thread tm2([&]() {
        add_AVX(a21, a22, r21, half_N);
        matMultiply(half_N, r21, b11, m2, args, argCount);
    });

    std::thread tm3([&]() {
        subtract_AVX(b12, b22, r32, half_N);
        matMultiply(half_N, a11, r32, m3, args, argCount);
    });

    std::thread tm4([&]() {
        subtract_AVX(b21, b11, r42, half_N);
        matMultiply(half_N, a22, r42, m4, args, argCount);
    });

    std::thread tm5([&]() {
        add_AVX(a11, a12, r51, half_N);
        matMultiply(half_N, r51, b22, m5, args, argCount);
    });

    std::thread tm6([&]() {
        subtract_AVX(a21, a11, r61, half_N);
        add_AVX(b11, b12, r62, half_N);
        matMultiply(half_N, r61, r62, m6, args, argCount);
    });

    std::thread tm7([&]() {
        subtract_AVX(a12, a22, r71, half_N);
        add_AVX(b21, b22, r72, half_N);
        matMultiply(half_N, r71, r72, m7, args, argCount);
    });

    tm1.join();
    tm2.join();
    tm3.join();
    tm4.join();
    tm5.join();
    tm6.join();
    tm7.join();

    float *temp11, *temp12, *temp21;
    float *temp31, *temp41, *temp42;
    temp11 = new float[half_N * half_N]();
    temp12 = new float[half_N * half_N]();
    temp21 = new float[half_N * half_N]();
    temp31 = new float[half_N * half_N]();
    temp41 = new float[half_N * half_N]();
    temp42 = new float[half_N * half_N]();
    gc.insert(gc.end(), {temp11, temp12, temp21, temp31, temp41, temp42});

    std::thread tc1([&]() {
        add_AVX(m1, m4, temp11, half_N);
        subtract_AVX(temp11, m5, temp12, half_N);
        add_AVX(temp12, m7, temp11, half_N);
        for (int i = 0; i < half_N; i++)
            for (int j = 0; j < half_N; j++)
                C[i * N + j] = temp11[i * half_N + j];
    });

    std::thread tc2([&]() {
        add_AVX(m3, m5, temp21, half_N);
        for (int i = 0; i < half_N; i++)
            for (int j = 0; j < half_N; j++)
                C[i * N + j + half_N] = temp21[i * half_N + j];
    });

    std::thread tc3([&]() {
        add_AVX(m2, m4, temp31, half_N);
        for (int i = 0; i < half_N; i++)
            for (int j = 0; j < half_N; j++)
                C[(i + half_N) * N + j] = temp31[i * half_N + j];
    });

    std::thread tc4([&]() {
        subtract_AVX(m1, m2, temp41, half_N);
        add_AVX(temp41, m3, temp42, half_N);
        add_AVX(temp42, m6, temp41, half_N);
        for (int i = 0; i < half_N; i++)
            for (int j = 0; j < half_N; j++)
                C[(i + half_N) * N + j + half_N] = temp41[i * half_N + j];
    });

    tc1.join();
    tc2.join();
    tc3.join();
    tc4.join();

    // cleanup, free memory
    for (auto& a : gc)
    {
        delete[] a;
    }
}

int nextPowerOfTwo(int n) {
    return static_cast<int>(pow(2, ceil(log2(n))));
}

// The new function to handle matrix multiplication where N is not a power of 2
void matrixPadding(int N, const float* A, const float* B, float* C, int* args, int argCount) {
    //printf("paddedMatrixMultiply is called\n");
    int M = nextPowerOfTwo(N);

    if (M == N) {
        // No need for padding, directly use matrixMultiply
        matrixMultiply(N, A, B, C, args, argCount);
        return;
    }

    // Create padded matrices
    float *A_padded, *B_padded, *C_padded;
    // malloc and set to all 0
    A_padded = new float[M * M]();
    B_padded = new float[M * M]();
    C_padded = new float[M * M]();

    // Copy A and B into the top-left corner of A_padded and B_padded
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A_padded[i * M + j] = A[i * N + j];
            B_padded[i * M + j] = B[i * N + j];
        }
    }

    // Multiply the padded matrices
    matrixMultiply(M, A_padded, B_padded, C_padded, args, argCount);

    // Extract the resulting N x N matrix from the top-left corner of C_padded
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] = C_padded[i * M + j];
        }
    }

    // Cleanup
    delete[] A_padded;
    delete[] B_padded;
    delete[] C_padded;
}

#define TEST
#ifdef TEST

bool isNear(float a, float b, float epsilon = 1e-3) {
    return std::abs(a - b) < epsilon;
}

void testPerformance(int N) {
    auto *A = new float[N*N];
    auto *B = new float[N*N];
    auto *C1 = new float[N*N];
    auto *C2 = new float[N*N];

    // Populate A and B with random values
    for (int i = 0; i < N * N; i++) {
        A[i] = (rand() % 21) - 8;
        B[i] = (rand() % 21) - 8;
    }

    // Time matrixMultiply function
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiply(N, A, B, C1, nullptr, 0);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Test 1 for N = " << N << " took " << duration.count() << "ms" << std::endl;

    // Time naiveMultiply function
    start = std::chrono::high_resolution_clock::now();
    msize_mul(N, A, B, C2);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "naiveMultiply for N = " << N << " took " << duration.count() << "ms" << std::endl;

    // Validate results
    double sum = 0;
    for (int i = 0; i < N * N; i++) {
        //assert(isNear(C1[i], C2[i]));
        sum += (C1[i] - C2[i]);
    }

    printf("diff = %lf\n", sum);

    delete[] A;
    delete[] B;
    delete[] C1;
    delete[] C2;
}

int main() {
    testPerformance(32);  // Test with a 32x32 matrix
    testPerformance(1024);  // Test with a 64x64 matrix
    testPerformance(2001);
    //testPerformance(3000);
    return 0;
}

#endif