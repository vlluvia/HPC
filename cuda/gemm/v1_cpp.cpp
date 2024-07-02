//
// Created by 92571 on 2024/7/1.
//

#include "include/v1_cpp.h"

void gemm_cpp(float *a, float *b, float *c, int M, int N, int K) {

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {

            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[ k * N + n];
            }
            c[i * N + j] = sum;
        }
    }
}

