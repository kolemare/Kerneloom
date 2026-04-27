#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace
{

    void check_cuda(cudaError_t result, const char *message)
    {
        if (result != cudaSuccess)
        {
            throw std::runtime_error(
                std::string(message) + ": " + cudaGetErrorString(result));
        }
    }

    __global__ void matmul_cuda_kernel(
        const float *a,
        const float *b,
        float *c,
        int m,
        int n,
        int k)
    {
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        float sum = 0.0f;

        for (int inner = 0; inner < k; ++inner)
        {
            sum += a[row * k + inner] * b[inner * n + col];
        }

        c[row * n + col] = sum;
    }

}

namespace kl::detail
{

    void matmul_cuda(
        const float *a,
        const float *b,
        float *c,
        int m,
        int n,
        int k)
    {
        float *d_a = nullptr;
        float *d_b = nullptr;
        float *d_c = nullptr;

        const std::size_t bytes_a = static_cast<std::size_t>(m) * k * sizeof(float);
        const std::size_t bytes_b = static_cast<std::size_t>(k) * n * sizeof(float);
        const std::size_t bytes_c = static_cast<std::size_t>(m) * n * sizeof(float);

        check_cuda(cudaMalloc(&d_a, bytes_a), "cudaMalloc d_a failed");
        check_cuda(cudaMalloc(&d_b, bytes_b), "cudaMalloc d_b failed");
        check_cuda(cudaMalloc(&d_c, bytes_c), "cudaMalloc d_c failed");

        check_cuda(cudaMemcpy(d_a, a, bytes_a, cudaMemcpyHostToDevice), "cudaMemcpy a failed");
        check_cuda(cudaMemcpy(d_b, b, bytes_b, cudaMemcpyHostToDevice), "cudaMemcpy b failed");

        constexpr int block_size = 16;

        dim3 block(block_size, block_size);
        dim3 grid(
            (n + block_size - 1) / block_size,
            (m + block_size - 1) / block_size);

        matmul_cuda_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);

        check_cuda(cudaGetLastError(), "CUDA kernel launch failed");
        check_cuda(cudaDeviceSynchronize(), "CUDA kernel execution failed");

        check_cuda(cudaMemcpy(c, d_c, bytes_c, cudaMemcpyDeviceToHost), "cudaMemcpy c failed");

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

}