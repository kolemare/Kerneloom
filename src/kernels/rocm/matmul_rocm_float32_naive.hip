#include <kernels/rocm/matmul_rocm_float32_naive.hiph>

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

namespace kl
{

    namespace
    {

        void check_hip(hipError_t error, const char *message)
        {
            if (error != hipSuccess)
            {
                throw std::runtime_error(
                    std::string(message) + ": " + hipGetErrorString(error));
            }
        }

        __global__ void matmul_rocm_float32_kernel(
            const float *a,
            const float *b,
            float *c,
            std::size_t m,
            std::size_t k,
            std::size_t n)
        {
            const std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
            const std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;

            if (row >= m || col >= n)
            {
                return;
            }

            float sum = 0.0f;

            for (std::size_t p = 0; p < k; ++p)
            {
                sum += a[row * k + p] * b[p * n + col];
            }

            c[row * n + col] = sum;
        }

    }

    void matmul_rocm_float32_naive(const Tensor &a, const Tensor &b, Tensor &c)
    {
        const std::size_t m = a.shape()[0];
        const std::size_t k = a.shape()[1];
        const std::size_t n = b.shape()[1];

        const float *a_data = static_cast<const float *>(a.data());
        const float *b_data = static_cast<const float *>(b.data());
        float *c_data = static_cast<float *>(c.data());

        constexpr int block_size = 16;

        dim3 block(block_size, block_size);
        dim3 grid(
            static_cast<unsigned int>((n + block_size - 1) / block_size),
            static_cast<unsigned int>((m + block_size - 1) / block_size));

        hipLaunchKernelGGL(
            matmul_rocm_float32_kernel,
            grid,
            block,
            0,
            0,
            a_data,
            b_data,
            c_data,
            m,
            k,
            n);

        check_hip(hipGetLastError(), "ROCm matmul kernel launch failed");
        check_hip(hipDeviceSynchronize(), "ROCm matmul synchronization failed");
    }

}