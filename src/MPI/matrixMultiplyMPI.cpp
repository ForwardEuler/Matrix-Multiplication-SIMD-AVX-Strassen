/**
 * Author: Chuan Tian
 * Date: Sep 28, 2023
 * Student ID: 45108907
 */
#include "matrixMultiplyMPI.h"
#include <cstdlib> //for malloc
#include <vector>

namespace MPI_FUNC
{
typedef char int8;
typedef short int16;
typedef int int32;
typedef long long int64;

typedef unsigned char byte;
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

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
    explicit memPool(u64 size): block_size(size)
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
        if (curr_size >= block_size)
        {
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
        for(auto& a : initptr)
        {
            free(a);
        }
        initptr.clear();
        curr_size = 0;
    }
};

inline void add_AVX(const float* A, const float* B, float* C, int N) {
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

inline void subtract_AVX(const float* A, const float* B, float* C, int N) {
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

inline void fast_mul(const int N, const float* A, const float* B, float* C) {
    //float* B_transpose = (float*)_mm_malloc(N * N * sizeof(float), 32);
    float* B_transpose = new float[N * N];
    constexpr int block = 32;
    for (int i = 0; i < N; i += block) {
        for(int j = 0; j < N; ++j) {
            int i_plus_b;
            for(int b = 0; b < block && (i_plus_b = i + b) < N; ++b) {
                B_transpose[j * N + i_plus_b] = B[i_plus_b * N + j];
            }
        }
    }
    for (int ii = 0; ii < N; ii += block) {
        for (int jj = 0; jj < N; jj += block) {
            for (int i = ii; i < ii + block && i < N; i++) {
                if (i + 1 < N) {
                    _mm_prefetch((const int8*) &C[(i + 1) * N], _MM_HINT_T1);
                    _mm_prefetch((const int8*) &A[(i + 1) * N], _MM_HINT_T1);
                }
                for (int j = jj; j < jj + block && j < N; j++) {
                    int row_r = i * N;
                    int row_c = j * N;
                    __m256 sum = _mm256_setzero_ps();
                    int k;
                    for (k = 0; k <= N - 8; k += 8) {
                        __m256 a = _mm256_loadu_ps(&A[row_r + k]);
                        __m256 b = _mm256_loadu_ps(&B_transpose[row_c + k]);
                        // mul and add to the sum
                        sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
                    }
                    sum = _mm256_hadd_ps(sum, sum);
                    sum = _mm256_hadd_ps(sum, sum);
                    float result[8];
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
    //_mm_free(B_transpose);
    delete[] B_transpose;
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
    // at N<=512, naive matrix multiplication with avx is faster than Strassen's algorithm
    if (N <= 512) {
        fast_mul(N, A, B, C);
        return;
    }
    const int half_N = N / 2;

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

    memPool pool2(sizeof(float) * half_N * half_N * 11 + 4);
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
} // namespace MPI_FUNC

void matrixMultiply_MPI(int N, const float* A, const float* B, float* C, int* args, int argCount)
{
    using namespace MPI_FUNC;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

#define FORTRAN_ORDER
#ifdef FORTRAN_ORDER
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

    std::vector<float*> gc;
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
    gc.insert(gc.end(), {a11, a12, a21, a22, b11, b12, b21, b22});

    std::thread t1([&]() {
        for (int i = 0; i < half_N; i++) {
            for (int j = 0; j < half_N; j++) {
                a11[i * half_N + j] = A[i * N + j];
                b11[i * half_N + j] = B[i * N + j];
                a12[i * half_N + j] = A[i * N + j + half_N];
                b12[i * half_N + j] = B[i * N + j + half_N];
            }
        }
    });

    std::thread t2([&]() {
        for (int i = 0; i < half_N; i++) {
            for (int j = 0; j < half_N; j++) {
                a21[i * half_N + j] = A[(i + half_N) * N + j];
                b21[i * half_N + j] = B[(i + half_N) * N + j];
                a22[i * half_N + j] = A[(i + half_N) * N + j + half_N];
                b22[i * half_N + j] = B[(i + half_N) * N + j + half_N];
            }
        }
    });

    t1.join();
    t2.join();

    //memPool pool2(sizeof(float) * half_N * half_N * 11);
    float *c11, *c12, *c21, *c22;
    c11 = new float[half_N * half_N];
    c12 = new float[half_N * half_N];
    c21 = new float[half_N * half_N];
    c22 = new float[half_N * half_N];
    float *r11 = new float [half_N * half_N];
    float *r12 = new float [half_N * half_N];
    float *r21 = new float [half_N * half_N];
    float *r22 = new float [half_N * half_N];
    gc.insert(gc.end(), {c11, c12, c21, c22});
    gc.insert(gc.end(), {r11, r12, r21, r22});

    if (rank == 0) {
        MPI_Request requests[4];

        std::thread tm1([&]() {
            matMultiply(half_N, a11, b11, r11, args, argCount);
        });
        std::thread tm2([&]() {
            matMultiply(half_N, a12, b21, r12, args, argCount);
        });
        std::thread tm3([&]() {
            matMultiply(half_N, a11, b12, r21, args, argCount);
        });
        std::thread tm4([&]() {
            matMultiply(half_N, a12, b22, r22, args, argCount);
        });

        tm1.join(); tm2.join();
        tm3.join(); tm4.join();

        add_AVX(r11, r12, c11, half_N);
        add_AVX(r21, r22, c12, half_N);

        MPI_Isend(c11, half_N * half_N, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, &requests[0]);
        MPI_Isend(c12, half_N * half_N, MPI_FLOAT, 1, 2, MPI_COMM_WORLD, &requests[1]);

        MPI_Irecv(c21, half_N * half_N, MPI_FLOAT, 1, 3, MPI_COMM_WORLD, &requests[2]);
        MPI_Irecv(c22, half_N * half_N, MPI_FLOAT, 1, 4, MPI_COMM_WORLD, &requests[3]);

        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
    }

    if (rank == 1) {
        MPI_Request requests[4];

        std::thread tm1([&]() {
            matMultiply(half_N, a21, b11, r11, args, argCount);
        });
        std::thread tm2([&]() {
            matMultiply(half_N, a22, b21, r12, args, argCount);
        });
        std::thread tm3([&]() {
            matMultiply(half_N, a21, b12, r21, args, argCount);
        });
        std::thread tm4([&]() {
            matMultiply(half_N, a22, b22, r22, args, argCount);
        });

        tm1.join(); tm2.join();
        tm3.join(); tm4.join();

        add_AVX(r11, r12, c21, half_N);
        add_AVX(r21, r22, c22, half_N);

        MPI_Isend(c21, half_N * half_N, MPI_FLOAT, 0, 3, MPI_COMM_WORLD, &requests[0]);
        MPI_Isend(c22, half_N * half_N, MPI_FLOAT, 0, 4, MPI_COMM_WORLD, &requests[1]);

        MPI_Irecv(c11, half_N * half_N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &requests[2]);
        MPI_Irecv(c12, half_N * half_N, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &requests[3]);

        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
    }

    // copy c11, c12, c21, c22 to C
    std::thread t3([&]() {
        for (int i = 0; i < half_N; i++) {
            for (int j = 0; j < half_N; j++) {
                C[i * N + j] = c11[i * half_N + j];
                C[i * N + j + half_N] = c12[i * half_N + j];
            }
        }
    });

    std::thread t4([&]() {
        for (int i = 0; i < half_N; i++) {
            for (int j = 0; j < half_N; j++) {
                C[(i + half_N) * N + j] = c21[i * half_N + j];
                C[(i + half_N) * N + j + half_N] = c22[i * half_N + j];
            }
        }
    });

    t3.join();
    t4.join();

    for (auto& a : gc) {
        delete[] a;
    }
}
