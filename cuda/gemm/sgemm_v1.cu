//
// Created by 92571 on 2024/7/1.
//
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include "include/sgemm_v1.cuh"
#include "v1_cpp.h"

#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

using namespace std;

float testError(
        void (*gpuSgemm)(float *, float *, float *, const int, const int, const int),
        dim3 gridDim, dim3 blockDim, const int M, const int N, const int K);

float testPerformance(
        void (*gpuSgemm)(float *, float *, float *, const int, const int, const int),
        dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat);

__global__ void sgemm_v1(float *a, float *b, float *c, int M, int N, int K) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    float r_c[TM][TN] = {0.0f};

    int load_a_smem_m = bx >> 1;
    int load_a_smem_k = (bx & 1) << 2;
    int load_b_smem_k = by >> 5;
    int load_b_smem_n = (by & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    for (int bk = 0; bk < (BK + K - 1) / BK; bk++) {

        int load_a_gmem_k = bx * BK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        s_a[load_a_smem_m][load_a_smem_k] = a[load_a_gmem_addr];
        int load_b_gmem_k = bx * BK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        s_b[load_b_smem_k][load_b_smem_n] = b[load_b_gmem_addr];

        for (int k = 0; k < BK; k++) {
            for (int m = 0; m < TM; m++) {
                for (int n = 0; n < TN; n++) {
                    int comp_c_smem_m = ty * TM + m;
                    int comp_c_smem_n = tx * TN + n;
                    r_c[m][n] += s_a[comp_c_smem_m][k] * s_b[k][comp_c_smem_n];
                }
            }
        }
        __syncthreads();

    }

    for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n++) {
            int load_c_gmem_m = by * BM + ty * TM + m;
            int load_c_gmem_n = bx * BN + tx * TN + n;
            int load_c_gmem_addr = load_c_gmem_m * N + load_c_gmem_n;
            c[load_c_gmem_addr] = r_c[m][n];
        }
    }

}

void main_c() {
    printf("\nKernal = sgemm_V1\n");
    const int outer_repeat = 10, inner_repeat = 1;
    const int BM = 128, BN = 128, TM = 8, TN = 8;
    void (*gpuSgemm)(float *, float *, float *, const int, const int, const int) = sgemm_v1;

    {
        const int M = 512, N = 512, K = 512;
        dim3 blockDim(BN / TN, BM / TM);
        dim3 gridDim((BN + N - 1) / BN, (BM + M - 1) / BM);
        float max_error = testError(gpuSgemm, gridDim, blockDim, M, N, K);
        printf("Max Error = %f\n", max_error);
    }

    const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};

    const int TESTNUM = 15;

    for (int i = 0; i < TESTNUM; i++) {
        const int M = M_list[i], N = N_list[i], K = K_list[i];

        dim3 blockDim(BN / TN, BM / TM);
        dim3 gridDim((BN + N - 1) / BN, (BM + M - 1) / BM);
        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;
        for (int j = 0; j < outer_repeat; j++) {
            double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }
        double avg_sec = total_sec / outer_repeat;
        double avg_Gflops = ((double) M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

        printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K,
               min_sec, avg_sec, max_sec, avg_Gflops);

    }

}

float testError(
        void (*gpuSgemm)(float *, float *, float *, const int, const int, const int),
        dim3 gridDim, dim3 blockDim, const int M, const int N, const int K) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;
    h_a = (float *) malloc(size_a);
    h_b = (float *) malloc(size_b);
    h_c = (float *) malloc(size_c);

    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (float *) malloc(size_c);
    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = rand() / float(RAND_MAX);
    for (int i = 0; i < K * N; i++)
        h_b[i] = rand() / float(RAND_MAX);
    cudaMemset(d_c, 15, size_c);

    gemm_cpp(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);
    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs(h_d_c[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d_c);
    return max_error;

}

float testPerformance(
        void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
        dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return sec;

}

