#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

__global__ void softmax_kernel(const float* __restrict__ input, float* __restrict__ output, int cols) {
    extern __shared__ float shared[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Load input to shared memory
    float val = -INFINITY;
    if (tid < cols) {
        val = input[row * cols + tid];
        shared[tid] = val;
    }
    __syncthreads();

    // Compute max
    float max_val = -INFINITY;
    for (int i = 0; i < cols; ++i)
        max_val = fmaxf(max_val, shared[i]);

    // Compute exp
    float sum_exp = 0.0f;
    for (int i = 0; i < cols; ++i) {
        shared[i] = expf(shared[i] - max_val);
        sum_exp += shared[i];
    }

    // Normalize
    if (tid < cols) {
        output[row * cols + tid] = shared[tid] / sum_exp;
    }
}
