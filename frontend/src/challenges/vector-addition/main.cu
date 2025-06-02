int main() {
    const int N = 1 << 20;
    const size_t size = N * sizeof(float);

    // Allocate host memory
    std::vector<float> h_A(N, 1.0f);
    std::vector<float> h_B(N, 2.0f);
    std::vector<float> h_C(N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy inputs to device
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back
    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    // Verify and print some results
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_C[i] - 3.0f) > 1e-5) {
            correct = false;
            break;
        }
    }

    std::cout << (correct ? "Vector addition PASSED\n" : "Vector addition FAILED\n");

    // Print first 10 results for verification
    std::cout << "Sample results:\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << "C[" << i << "] = " << h_C[i] << std::endl;
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}