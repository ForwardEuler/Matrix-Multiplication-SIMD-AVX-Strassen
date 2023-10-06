/**
 * Author: Chuan Tian
 * Date: Sep 28, 2023
 * Student ID: 45108907
 */
#include "matrixMultiply.h"
#include <cstdlib> //for malloc
#include <immintrin.h>
#include <thread>
#include <vector>

typedef char int8;
typedef short int16;
typedef int int32;
typedef long long int64;

typedef unsigned char byte;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

// Uncomment the line below for standalone test on local machine
// #define LOCALTEST
#ifndef LOCALTEST
#define FORTRAN_ORDER
#endif

/**
 * @brief WARNING: this class is not thread safe
 */
class memPool
{
    u64 curr_size;
    byte* stack_ptr;
    std::vector<byte*> initptr;
    u64 block_size;

  public:
    /**
     * @brief WARNING: this class is not thread safe
     * @param size size of each block in memPool in bytes, pool will automatically expand by this size if full
     */
    explicit memPool(u64 size) : block_size(size)
    {
        byte* tmp = (byte*)malloc(size);
        stack_ptr = tmp;
        initptr.push_back(tmp);
        curr_size = 0;
    }

    ~memPool()
    {
        memfree();
    }

    /**
     * @brief allocate memory from pool, pool will automatically expand if full
     * @param size size of memory in bytes to allocate
     * @return void* pointer to the allocated memory (like malloc)
     */
    void* memloc(u64 size)
    {
        curr_size += size;
        block_size = size > block_size ? size : block_size;
        if (curr_size >= block_size) {
            byte* tmp = (byte*)malloc(block_size);
            initptr.push_back(tmp);
            curr_size = size;
            stack_ptr = tmp + size;
            return tmp;
        }
        byte* ret_ptr = stack_ptr;
        stack_ptr += size;
        return (void*)ret_ptr;
    }

    /**
     * @brief free all memory in memPool
     */
    void memfree()
    {
        for (auto& a : initptr) {
            free(a);
        }
        initptr.clear();
        curr_size = 0;
    }
};

void mat_pad_mul(int N, const float* A, const float* B, float* C);

inline void add_AVX(const float* A, const float* B, float* C, int N)
{
    const int l = N * N;
    const int ub = l - 8;
    int i;
    // add eight floats at a time
    for (i = 0; i <= ub; i += 8) {
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

inline void subtract_AVX(const float* A, const float* B, float* C, int N)
{
    const int l = N * N;
    const int ub = l - 8;
    int i;
    // subtract eight floats at a time
    for (i = 0; i <= ub; i += 8) {
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

inline void strassenMultiply(const float A[4], const float B[4], float C[4])
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

inline void fast_mul(const int N, const float* A, const float* B, float* C)
{
    auto* B_transpose = (float*)_mm_malloc(N * N * sizeof(float), 32);
    // float* B_transpose = new float[N * N];
    constexpr int block = 32;
    for (int i = 0; i < N; i += block) {
        for (int j = 0; j < N; ++j) {
            int i_plus_b;
            for (int b = 0; b < block && (i_plus_b = i + b) < N; ++b) {
                B_transpose[j * N + i_plus_b] = B[i_plus_b * N + j];
            }
        }
    }
    for (int ii = 0; ii < N; ii += block) {
        for (int jj = 0; jj < N; jj += block) {
            for (int i = ii; i < ii + block && i < N; i++) {
                if (i + 1 < N) {
                    _mm_prefetch((const int8*)&C[(i + 1) * N], _MM_HINT_T1);
                    _mm_prefetch((const int8*)&A[(i + 1) * N], _MM_HINT_T1);
                }
                for (int j = jj; j < jj + block && j < N; j++) {
                    int row_r = i * N;
                    int row_c = j * N;
                    alignas(block) __m256 sum = _mm256_setzero_ps();
                    int k;
                    for (k = 0; k <= N - 8; k += 8) {
                        __m256 a = _mm256_loadu_ps(&A[row_r + k]);
                        __m256 b = _mm256_loadu_ps(&B_transpose[row_c + k]);
                        // mul and add to the sum
                        sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
                    }
                    sum = _mm256_hadd_ps(sum, sum);
                    sum = _mm256_hadd_ps(sum, sum);
                    alignas(32) float result[8];
                    _mm256_storeu_ps(result, sum);
                    C[row_r + j] = result[0] + result[4];
                    // Handle the case when N is not a multiple of 8
                    for (; k < N; k++) {
                        C[row_r + j] += A[row_r + k] * B_transpose[row_c + k];
                    }
                }
            }
        }
    }
    _mm_free(B_transpose);
    // delete[] B_transpose;
}

void matMultiply(int N, const float* A, const float* B, float* C)
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
    // at N<=256 or 512, naive matrix multiplication with avx is faster than Strassen's algorithm
    if (N <= 512 || N % 2 != 0) {
        fast_mul(N, A, B, C);
        return;
    }

    const int half_N = N / 2;
    float *a11, *a12, *a21, *a22;
    float *b11, *b12, *b21, *b22;
    memPool pool1(sizeof(float) * half_N * half_N * 8 + 4);
    u64 alloc_size = sizeof(float) * half_N * half_N;
    a11 = (float*)pool1.memloc(alloc_size);
    a12 = (float*)pool1.memloc(alloc_size);
    a21 = (float*)pool1.memloc(alloc_size);
    a22 = (float*)pool1.memloc(alloc_size);

    b11 = (float*)pool1.memloc(alloc_size);
    b12 = (float*)pool1.memloc(alloc_size);
    b21 = (float*)pool1.memloc(alloc_size);
    b22 = (float*)pool1.memloc(alloc_size);

    for (int i = 0; i < half_N; i++) {
        for (int j = 0; j < half_N; j++) {
            a11[i * half_N + j] = A[i * N + j];
            a12[i * half_N + j] = A[i * N + j + half_N];
            a21[i * half_N + j] = A[(i + half_N) * N + j];
            a22[i * half_N + j] = A[(i + half_N) * N + j + half_N];
        }
    }
    for (int i = 0; i < half_N; i++) {
        for (int j = 0; j < half_N; j++) {
            b11[i * half_N + j] = B[i * N + j];
            b12[i * half_N + j] = B[i * N + j + half_N];
            b21[i * half_N + j] = B[(i + half_N) * N + j];
            b22[i * half_N + j] = B[(i + half_N) * N + j + half_N];
        }
    }

    memPool pool2(sizeof(float) * half_N * half_N * 11 + 4);
    float *m1, *m2, *m3, *m4, *m5, *m6, *m7;
    m1 = (float*)pool2.memloc(alloc_size);
    m2 = (float*)pool2.memloc(alloc_size);
    m3 = (float*)pool2.memloc(alloc_size);
    m4 = (float*)pool2.memloc(alloc_size);
    m5 = (float*)pool2.memloc(alloc_size);
    m6 = (float*)pool2.memloc(alloc_size);
    m7 = (float*)pool2.memloc(alloc_size);

    float *r1, *r2;
    r1 = (float*)pool2.memloc(alloc_size);
    r2 = (float*)pool2.memloc(alloc_size);

    // M1 = (A11 + A22) * (B11 + B22)
    add_AVX(a11, a22, r1, half_N);
    add_AVX(b11, b22, r2, half_N);
    matMultiply(half_N, r1, r2, m1);
    // M2 = (A21 + A22) * B11
    add_AVX(a21, a22, r1, half_N);
    matMultiply(half_N, r1, b11, m2);
    // M3 = A11 * (B12 - B22)
    subtract_AVX(b12, b22, r2, half_N);
    matMultiply(half_N, a11, r2, m3);
    // M4 = A22 * (B21 - B11)
    subtract_AVX(b21, b11, r2, half_N);
    matMultiply(half_N, a22, r2, m4);
    // M5 = (A11 + A12) * B22
    add_AVX(a11, a12, r1, half_N);
    matMultiply(half_N, r1, b22, m5);
    // M6 = (A21 - A11) * (B11 + B12)
    subtract_AVX(a21, a11, r1, half_N);
    add_AVX(b11, b12, r2, half_N);
    matMultiply(half_N, r1, r2, m6);
    // M7 = (A12 - A22) * (B21 + B22)
    subtract_AVX(a12, a22, r1, half_N);
    add_AVX(b21, b22, r2, half_N);
    matMultiply(half_N, r1, r2, m7);

    float *temp1, *temp2;
    temp1 = (float*)pool2.memloc(alloc_size);
    temp2 = (float*)pool2.memloc(alloc_size);

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
 * @param[in] args : pointer to array of integers which can be used for debugging and performance tweaks. Optional. If
 * unused, set to zero
 * @param[in] argCount : the length of the flags array
 * @return void
 * */
void matrixMultiply(int N, const float* A, const float* B, float* C, int* args, int argCount)
{
#ifdef FORTRAN_ORDER
    // Change to fortran order, due to transpose law
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

    if (N <= 512) {
        fast_mul(N, A, B, C);
        return;
    }

    if (N % 2 != 0) {
        mat_pad_mul(N, A, B, C);
        return;
    }

    float *a11, *a12, *a21, *a22;
    float *b11, *b12, *b21, *b22;
    const u64 alloc_size = sizeof(float) * half_N * half_N;
    memPool pool1(alloc_size * 8 + 4);

    a11 = (float*)pool1.memloc(alloc_size);
    a12 = (float*)pool1.memloc(alloc_size);
    a21 = (float*)pool1.memloc(alloc_size);
    a22 = (float*)pool1.memloc(alloc_size);

    b11 = (float*)pool1.memloc(alloc_size);
    b12 = (float*)pool1.memloc(alloc_size);
    b21 = (float*)pool1.memloc(alloc_size);
    b22 = (float*)pool1.memloc(alloc_size);

    std::thread t1([&]() {
        for (int i = 0; i < half_N; i++) {
            for (int j = 0; j < half_N; j++) {
                a11[i * half_N + j] = A[i * N + j];
                a12[i * half_N + j] = A[i * N + j + half_N];
                a21[i * half_N + j] = A[(i + half_N) * N + j];
                a22[i * half_N + j] = A[(i + half_N) * N + j + half_N];
            }
        }
    });

    std::thread t2([&]() {
        for (int i = 0; i < half_N; i++) {
            for (int j = 0; j < half_N; j++) {
                b11[i * half_N + j] = B[i * N + j];
                b12[i * half_N + j] = B[i * N + j + half_N];
                b21[i * half_N + j] = B[(i + half_N) * N + j];
                b22[i * half_N + j] = B[(i + half_N) * N + j + half_N];
            }
        }
    });

    t1.join();
    t2.join();

    float *m1, *m2, *m3, *m4, *m5, *m6, *m7;
    memPool pool2(alloc_size * 7 + 4);
    m1 = (float*)pool2.memloc(alloc_size);
    m2 = (float*)pool2.memloc(alloc_size);
    m3 = (float*)pool2.memloc(alloc_size);
    m4 = (float*)pool2.memloc(alloc_size);
    m5 = (float*)pool2.memloc(alloc_size);
    m6 = (float*)pool2.memloc(alloc_size);
    m7 = (float*)pool2.memloc(alloc_size);

    float *r11, *r12, *r21, *r32, *r42;
    float *r51, *r61, *r62, *r71, *r72;

    memPool pool3(alloc_size * 10 + 4);
    r11 = (float*)pool3.memloc(alloc_size);
    r12 = (float*)pool3.memloc(alloc_size);
    r21 = (float*)pool3.memloc(alloc_size);
    r32 = (float*)pool3.memloc(alloc_size);
    r42 = (float*)pool3.memloc(alloc_size);
    r51 = (float*)pool3.memloc(alloc_size);
    r61 = (float*)pool3.memloc(alloc_size);
    r62 = (float*)pool3.memloc(alloc_size);
    r71 = (float*)pool3.memloc(alloc_size);
    r72 = (float*)pool3.memloc(alloc_size);

    std::thread tm1([&]() {
        add_AVX(a11, a22, r11, half_N);
        add_AVX(b11, b22, r12, half_N);
        matMultiply(half_N, r11, r12, m1);
    });

    std::thread tm2([&]() {
        add_AVX(a21, a22, r21, half_N);
        matMultiply(half_N, r21, b11, m2);
    });

    std::thread tm3([&]() {
        subtract_AVX(b12, b22, r32, half_N);
        matMultiply(half_N, a11, r32, m3);
    });

    std::thread tm4([&]() {
        subtract_AVX(b21, b11, r42, half_N);
        matMultiply(half_N, a22, r42, m4);
    });

    std::thread tm5([&]() {
        add_AVX(a11, a12, r51, half_N);
        matMultiply(half_N, r51, b22, m5);
    });

    std::thread tm6([&]() {
        subtract_AVX(a21, a11, r61, half_N);
        add_AVX(b11, b12, r62, half_N);
        matMultiply(half_N, r61, r62, m6);
    });

    std::thread tm7([&]() {
        subtract_AVX(a12, a22, r71, half_N);
        add_AVX(b21, b22, r72, half_N);
        matMultiply(half_N, r71, r72, m7);
    });

    tm1.join(); tm2.join(); tm3.join();
    tm4.join(); tm5.join(); tm6.join();
    tm7.join();

    float *temp11, *temp12, *temp21;
    float *temp31, *temp41, *temp42;

    memPool pool4(alloc_size * 6 + 4);
    temp11 = (float*)pool4.memloc(alloc_size);
    temp12 = (float*)pool4.memloc(alloc_size);
    temp21 = (float*)pool4.memloc(alloc_size);
    temp31 = (float*)pool4.memloc(alloc_size);
    temp41 = (float*)pool4.memloc(alloc_size);
    temp42 = (float*)pool4.memloc(alloc_size);

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

    pool1.memfree();
    pool2.memfree();
    pool3.memfree();
    pool4.memfree();
}

void mat_pad_mul(int N, const float* A, const float* B, float* C)
{
    // printf("paddedMatrixMultiply is called\n");
    int M = N + 1;

    float *A_padded, *B_padded, *C_padded;
    // malloc and set to all 0
    A_padded = new float[M * M]();
    B_padded = new float[M * M]();
    C_padded = new float[M * M];

    // Copy A and B into the top-left corner of A_padded and B_padded
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A_padded[i * M + j] = A[i * N + j];
            B_padded[i * M + j] = B[i * N + j];
        }
    }

#ifndef FORTRAN_ORDER
    matrixMultiply(M, A_padded, B_padded, C_padded, nullptr, 0);
#else
    matrixMultiply(M, B_padded, A_padded, C_padded, nullptr, 0);
#endif

    // Extract the resulting N x N matrix from the top-left corner of C_padded
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] = C_padded[i * M + j];
        }
    }

    delete[] A_padded;
    delete[] B_padded;
    delete[] C_padded;
}

#ifdef LOCALTEST

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

void thread_multi(const int start, const int end, const int N, const float* A, const float* B_transpose, float* C)
{
    constexpr int block = 32;
    for (int ii = start; ii < end; ii += block) {
        for (int jj = 0; jj < N; jj += block) {
            for (int i = ii; i < ii + block && i < N; i++) {
                if (i + 1 < N) {
                    _mm_prefetch((const int8*)&C[(i + 1) * N], _MM_HINT_T1);
                    _mm_prefetch((const int8*)&A[(i + 1) * N], _MM_HINT_T1);
                }
                for (int j = jj; j < jj + block && j < N; j++) {
                    int row_r = i * N;
                    int row_c = j * N;
                    alignas(block) __m256 sum = _mm256_setzero_ps();
                    int k;
                    for (k = 0; k <= N - 8; k += 8) {
                        __m256 a = _mm256_loadu_ps(&A[row_r + k]);
                        __m256 b = _mm256_loadu_ps(&B_transpose[row_c + k]);
                        sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
                    }
                    sum = _mm256_hadd_ps(sum, sum);
                    sum = _mm256_hadd_ps(sum, sum);
                    alignas(32) float result[8];
                    _mm256_storeu_ps(result, sum);
                    C[row_r + j] = result[0] + result[4];
                    for (; k < N; k++) {
                        C[row_r + j] += A[row_r + k] * B_transpose[row_c + k];
                    }
                }
            }
        }
    }
}

inline void general_mul(const int N, const float* A, const float* B, float* C)
{
    auto* B_transpose = (float*)_mm_malloc(N * N * sizeof(float), 32);

    constexpr int block = 32;
    for (int i = 0; i < N; i += block) {
        for (int j = 0; j < N; ++j) {
            int i_plus_b;
            for (int b = 0; b < block && (i_plus_b = i + b) < N; ++b) {
                B_transpose[j * N + i_plus_b] = B[i_plus_b * N + j];
            }
        }
    }

    std::vector<std::thread> threads;
    int num_threads = 8;
    int chunk_size = N / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk_size;
        int end = (t == num_threads - 1) ? N : start + chunk_size; // To handle the last chunk
        threads.emplace_back(thread_multi, start, end, N, A, B_transpose, C);
    }

    for (auto& t : threads) {
        t.join();
    }

    _mm_free(B_transpose);
}

void testPerformance(int N)
{
    auto* A = new float[N * N];
    auto* B = new float[N * N];
    auto* C1 = new float[N * N];
    auto* C2 = new float[N * N];

    std::random_device rd;
    std::mt19937 gen(rd());

    // Create a uniform real distribution between -10 and 10
    std::uniform_real_distribution<> dis(-10.0, 10.0);

    for (int i = 0; i < N * N; i++) {
        A[i] = (float)dis(gen);
        B[i] = (float)dis(gen);
    }

    // Time matrixMultiply function
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiply(N, A, B, C1, nullptr, 0);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Test 1 for N = " << N << " took " << duration.count() << "ms" << std::endl;

    // Time naiveMultiply function
    start = std::chrono::high_resolution_clock::now();
    general_mul(N, A, B, C2);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "reference Multiply for N = " << N << " took " << duration.count() << "ms" << std::endl;

    // Validate result using RMSE
    float sse = 0;
    for (int i = 0; i < N * N; i++) {
        float error = C1[i] - C2[i];
        sse += error * error;
    }
    float rmse = sqrt(sse / (N * N));

    printf("RMSE err = %f\n", rmse);

    delete[] A;
    delete[] B;
    delete[] C1;
    delete[] C2;
}

int main()
{
    testPerformance(4321);
    testPerformance(4096);
    return 0;
}

#endif