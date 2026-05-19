#include <kernels/cuda/activation/backward_tanh_cuda_float32.cuh>

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace kl
{

    namespace
    {

        void check_cuda(cudaError_t error, const char *message)
        {
            if (error != cudaSuccess)
            {
                throw std::runtime_error(
                    std::string(message) + ": " + cudaGetErrorString(error));
            }
        }

        __global__ void backward_tanh_cuda_float32_kernel(
            const float *activation_output,
            float *grad,
            std::size_t count)
        {
            const std::size_t index =
                blockIdx.x * blockDim.x + threadIdx.x;

            if (index >= count)
            {
                return;
            }

            const float y = activation_output[index];

            grad[index] *= 1.0f - y * y;
        }

    }

    void backward_tanh_cuda_float32(
        const Tensor &activation_output,
        Tensor &grad)
    {
        const float *activation_output_data =
            static_cast<const float *>(activation_output.data());

        float *grad_data =
            static_cast<float *>(grad.data());

        const std::size_t count = grad.numel();

        constexpr int block_size = 256;

        dim3 block(block_size);
        dim3 grid(static_cast<unsigned int>((count + block_size - 1) / block_size));

        backward_tanh_cuda_float32_kernel<<<grid, block>>>(
            activation_output_data,
            grad_data,
            count);

        check_cuda(cudaGetLastError(), "CUDA Tanh backward kernel launch failed");
        check_cuda(cudaDeviceSynchronize(), "CUDA Tanh backward synchronization failed");
    }

}