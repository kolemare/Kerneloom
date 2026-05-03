#include <kernels/rocm/conv2d_rocm_float32_naive.hiph>

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

        __global__ void conv2d_rocm_float32_kernel(
            const float *input,
            const float *kernels,
            float *result,
            std::size_t N,
            std::size_t C,
            std::size_t H,
            std::size_t W,
            std::size_t K,
            std::size_t R,
            std::size_t S,
            std::size_t OH,
            std::size_t OW)
        {
            const std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
            const std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;
            const std::size_t n = blockIdx.z / K;
            const std::size_t k = blockIdx.z % K;

            if (row >= OH || col >= OW)
            {
                return;
            }

            float sum = 0.0f;

            for (std::size_t c = 0; c < C; c++)
            {
                for (std::size_t r = 0; r < R; r++)
                {
                    for (std::size_t s = 0; s < S; s++)
                    {
                        sum += input[n * C * H * W + c * H * W + (row + r) * W + (col + s)] *
                               kernels[k * C * R * S + c * R * S + r * S + s];
                    }
                }
            }

            result[n * K * OH * OW + k * OH * OW + row * OW + col] = sum;
        }

    }

    void conv2d_rocm_float32_naive(const Tensor &input, const Tensor &kernels, Tensor &result)
    {
        const std::size_t N = input.shape()[0];
        const std::size_t C = input.shape()[1];
        const std::size_t H = input.shape()[2];
        const std::size_t W = input.shape()[3];

        const std::size_t K = kernels.shape()[0];
        const std::size_t R = kernels.shape()[2];
        const std::size_t S = kernels.shape()[3];

        const std::size_t OH = H - R + 1;
        const std::size_t OW = W - S + 1;

        const float *input_data = static_cast<const float *>(input.data());
        const float *kernel_data = static_cast<const float *>(kernels.data());
        float *result_data = static_cast<float *>(result.data());

        constexpr int block_size = 16;

        dim3 block(block_size, block_size);
        dim3 grid(
            static_cast<unsigned int>((OW + block_size - 1) / block_size),
            static_cast<unsigned int>((OH + block_size - 1) / block_size),
            static_cast<unsigned int>(N * K));

        hipLaunchKernelGGL(
            conv2d_rocm_float32_kernel,
            grid,
            block,
            0,
            0,
            input_data,
            kernel_data,
            result_data,
            N,
            C,
            H,
            W,
            K,
            R,
            S,
            OH,
            OW);

        check_hip(hipGetLastError(), "ROCm conv2d kernel launch failed");
        check_hip(hipDeviceSynchronize(), "ROCm conv2d synchronization failed");
    }

}