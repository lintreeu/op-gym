int main() {
    // Matrix dimensions
    const int N = 512; // rows of A and C
    const int K = 256; // cols of A, rows of B
    const int M = 384; // cols of B and C

    size_t size_A = N * K * sizeof(float);
    size_t size_B = K * M * sizeof(float);
    size_t size_C = N * M * sizeof(float);

    // Host allocation
    std::vector<float> h_A(N * K, 1.0f);
    std::vector<float> h_B(K * M, 2.0f);
    std::vector<float> h_C(N * M);

    // Device allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice);

    // Launch parameters
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Kernel launch
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N, K, M);

    cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost);

    // Simple verification
    float expected = K * 2.0f;
    bool correct = true;
    for (int i = 0; i < N * M; ++i) {
        if (fabs(h_C[i] - expected) > 1e-5f) {
            correct = false;
            break;
        }
    }

    std::cout << (correct ? "Matrix multiplication PASSED\n" : "Matrix multiplication FAILED\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}