#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

namespace
{

    void check_hip(hipError_t result, const char *message)
    {
        if (result != hipSuccess)
        {
            throw std::runtime_error(
                std::string(message) + ": " + hipGetErrorString(result));
        }
    }

    __global__ void matmul_rocm_kernel(
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

    void matmul_rocm(
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

        check_hip(hipMalloc(&d_a, bytes_a), "hipMalloc d_a failed");
        check_hip(hipMalloc(&d_b, bytes_b), "hipMalloc d_b failed");
        check_hip(hipMalloc(&d_c, bytes_c), "hipMalloc d_c failed");

        check_hip(hipMemcpy(d_a, a, bytes_a, hipMemcpyHostToDevice), "hipMemcpy a failed");
        check_hip(hipMemcpy(d_b, b, bytes_b, hipMemcpyHostToDevice), "hipMemcpy b failed");

        constexpr int block_size = 16;

        dim3 block(block_size, block_size);
        dim3 grid(
            (n + block_size - 1) / block_size,
            (m + block_size - 1) / block_size);

        hipLaunchKernelGGL(
            matmul_rocm_kernel,
            grid,
            block,
            0,
            0,
            d_a,
            d_b,
            d_c,
            m,
            n,
            k);

        check_hip(hipGetLastError(), "ROCm kernel launch failed");
        check_hip(hipDeviceSynchronize(), "ROCm kernel execution failed");

        check_hip(hipMemcpy(c, d_c, bytes_c, hipMemcpyDeviceToHost), "hipMemcpy c failed");

        hipFree(d_a);
        hipFree(d_b);
        hipFree(d_c);
    }

}