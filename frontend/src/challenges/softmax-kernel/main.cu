int main() {
    const int rows = 2;
    const int cols = 4;

    std::vector<float> h_input = {
        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 1.0f, 0.0f, -1.0f
    };
    std::vector<float> h_output(rows * cols);

    float *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(float) * rows * cols);
    cudaMalloc(&d_output, sizeof(float) * rows * cols);

    cudaMemcpy(d_input, h_input.data(), sizeof(float) * rows * cols, cudaMemcpyHostToDevice);

    softmax_kernel<<<rows, cols, sizeof(float) * cols>>>(d_input, d_output, cols);

    cudaMemcpy(h_output.data(), d_output, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);

    for (int r = 0; r < rows; ++r) {
        std::cout << "Row " << r << ": ";
        for (int c = 0; c < cols; ++c) {
            std::cout << h_output[r * cols + c] << " ";
        }
        std::cout << "\n";
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}