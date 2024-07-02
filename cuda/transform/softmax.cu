//
// Created by 92571 on 2024/7/2.
//



void softmax_cpu(float *out, const float *inp, int N, int C) {
    for (int i = 0; i < N; i++) {
        const float *x = inp + i * C;
        float *y = out + i * C;

        float max_val = -INFINITY;
        for(int j = 0; j < C; j++){
            max_val = fmaxf(max_val, x[j]);
        }

        double sum = 0.0;
        for(int j = 0; j < C; j++){
            out[j] = expf(x[j] - max_val);
            sum += out[j];
        }
        for(int j = 0; j < C; j++){
            out[j] /= sum;
        }
    }
}


// ---------------------------------------------------
__global__ void softmax_forward_kernel1(float *out, float * inp, int N, int C){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < N){
        float * x = inp + idx * C;
        float * y = out + idx * C;

        float max_val = -INFINITY;
        for(int j = 0 ; j < C; j++){
            max_val = fmaxf(max_val, x[j]);
        }
        float  sum = 0.0f;
        for (int j = 0 ; j < C; j++) {
            y[j] = expf(x[j] - max_val);
            sum += y[j];
        }
        for (int j = 0 ; j < C; j++) {
            y[j] /= sum;
        }
    }
}

__global__  void softmax_forward_kernel2(float* out, const float* inp, int N, int C) {

    extern __shared__ float  shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    const float *x = inp + idx * C;
    float max_val = -INFINITY;
    for(int i = tid; i < C; i+=block_size){
        max_val = fmaxf(max_val, x[i]);
    }
    shared[tid] = max_val;
    __syncthreads();
    for(int stride = block_size / 2; stride > 0; stride /= 2){
        __syncthreads();
        if(tid < stride){
            shared[tid] = fmaxf(shared[tid] ,shared[tid + stride]);
        }
    }

    __syncthreads();
    float offset = shared[0];
    for(int i = tid;  i < C; i += block_size){
        out[idx * C +i] = expf(x[i] - offset);
    }

    __syncthreads();
    x = out + idx * C;
    float sumval  = 0.0f;
    for(int i = tid; i < C; i += block_size){
        sumval += x[i];
    }
    shared[tid] = sumval;
    __syncthreads();

    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // broadcast the sum to all threads in the block
    __syncthreads();
    float sum = shared[0];
    // divide the input values by the sum
    for (int i = tid; i < C; i += block_size) {
        out[idx * C + i] = x[i] / sum;
    }
}


