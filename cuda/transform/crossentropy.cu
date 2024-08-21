//
// Created by 92571 on 2024/7/2.
//


#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"

void crossentropy_cpu(float* losses,
                      const float* probs, const int* targets,
                      int B, int T, int V) {

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* probs_bt = probs + b * T * V + t * V;  // probs[b,t,:]
            int ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

__global__ void crossentropy_forward_kernel1(float* losses,
                                             const float* probs, const int* targets,
                                             int B, int T, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B * T) {
        int b = i / T;
        int t = i % T;
        const float* probs_bt = probs + b * T * V + t * V;
        int ix = targets[b * T + t];
        losses[b * T + t] = -logf(probs_bt[ix]);
    }
}
