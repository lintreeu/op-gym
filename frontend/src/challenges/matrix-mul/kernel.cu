#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// CUDA Kernel: C = A x B
__global__ void matmul_kernel(const float* A, const float* B, float* C, int N, int K, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row in C
    int col = blockIdx.x * blockDim.x + threadIdx.x; // col in C

    if (row < N && col < M) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * M + col];
        }
        C[row * M + col] = sum;
    }
}
