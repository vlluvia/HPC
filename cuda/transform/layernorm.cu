//
// Created by 92571 on 2024/7/2.
//

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"

void layernorm_forward_cpu(
        float *out, float *mean, float *rstd, const float *inp, const float *weight, const float *bias,
        int B, int T, int C
) {

    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float *x = inp + b * T * C + t * C;

            float m = 0.0f;
            for (int c = 0; c < C; c++) {
                m += x[c];
            }
            m /= C;

            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                int temp = x[i] - m;
                v += (temp * temp);
            }
            v /= C;

            float *y = out + b * T * C + t * C;
            float s = 1.0f / sqrtf(v - eps);
            for (int i = 0; i < C; i++) {
                float n = (x[i] - m) * s;
                float o = n * weight[i] + bias[i];
                y[i] = o;
            }
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

__global__ void layernorm_forward_gpu_v1(
        float *out, float *mean, float *rstd, const float *inp, const float *weight, const float *bias,
        int N, int C
) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    float eps = 1e-5f;
    if(idx < N){
        const float *x = inp + idx * C;
        float *y = out + idx * C;

        float m = 0.0f;
        for(int i = 0 ; i < C; i++){
            m += x[i];
        }
        m /= C;

        float v = 0.0f;
        for(int i = 0; i < C; i++){
            int temp = (x[i] - m);
            v += temp * temp;
        }
        v /= C;

        float s = 1.0f / sqrtf(v + eps);
        for(int i = 0; i < C; i++){
            int n = (x[i] - m) * s;
            int o = n * weight[i] + bias[i];
            y[i] = o;
        }
        mean[idx] = m;
        rstd[idx] = s;
    }
}


__global__ void mean_kernel(float *mean, const float* inp, int N, int C, int block_size){
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    const float *x = inp + idx * C;

    float sum = 0.0f;
    for(int i= tid ; i < C; i+=block_size){
        sum += x[i];
    }
    shared[tid] = sum;
    __syncthreads();
    for(int stride = block_size / 2 ; stride > 0; stride /= 2){
        __syncthreads();
        if(tid < stride){
            shared[tid] += shared[tid + stride];
        }
    }
    if(tid == 0){
        mean[idx] = shared[0] / C;
    }

}


__global__ void rstd_kernel(float *rstd, const float *inp, const float* mean,
                            int N, int C, int block_size){
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    const float *x = inp + idx * C;
    float m = shared[idx];

    float sum = 0.0f;
    for(int i = tid; i < C; i += block_size){
        int temp = x[i] - m;
        sum += temp* temp;
    }
    shared[tid] = sum;
    __syncthreads();
    for(int stride = block_size / 2; stride > 0; stride /= 2){
        __syncthreads();
        if(tid < stride){
            shared[tid] = shared[tid + stride];
        }
    }
    if(tid == 0){
        rstd[idx] = 1.0f / sqrtf(shared[0] / C - 1e-5f);
    }
}


__global__ void normalization_kernel(float* out, const float* inp, float* mean, float* rstd,
                                     const float* weight, const float* bias, int B, int T, int C){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int bt = idx /C;
    int c  = idx %C;

    float m = mean[bt];
    float s = rstd[bt];
    float xi = inp[idx];
    float n  = s * (xi - m);
    float o  = n * weight[c] + bias[c];
    out[idx] = o;
}

// ----------------------------------------------------------------------------
// kernel launcher
void layernorm_forward1(float* out, float* mean, float* rstd,
                        const float* inp, const float* weight, const float* bias,
                        int B, int T, int C,
                        const int block_size) {
    const int N = B * T;
    const int grid_size = ceil_div(N, block_size);
    layernorm_forward_gpu_v1<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

void layernorm_forward2(float *out, float *mean, float *rstd,
                        float *inp, const float * weight, float* bias,
                        int B, int T, int C, int block_size){
    int N = B * T;
    mean_kernel<<<N, block_size, block_size * sizeof(float )>>>(mean, inp, N, C, block_size);
    cudaCheck(cudaGetLastError());
    rstd_kernel<<<N, block_size, block_size * sizeof(float)>>>(rstd, inp, mean, N, C, block_size);
    cudaCheck(cudaGetLastError());

    const int block_size2  = 256;
    const int grid_size = ceil_div(B * T * C, block_size2);
    normalization_kernel<<<grid_size, block_size>>>(out, inp, mean, rstd, weight, bias, B, T, C);
    cudaCheck(cudaGetLastError());

}

