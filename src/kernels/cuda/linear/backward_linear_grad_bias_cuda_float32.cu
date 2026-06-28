#include <kernels/cuda/linear/backward_linear_grad_bias_cuda_float32.cuh>

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

        __global__ void backward_linear_grad_bias_cuda_float32_kernel(
            const float *grad_output,
            float *grad_bias,
            std::size_t batch_size,
            std::size_t output_features)
        {
            const std::size_t out =
                blockIdx.x * blockDim.x + threadIdx.x;

            if (out >= output_features)
            {
                return;
            }

            float sum = 0.0f;

            for (std::size_t n = 0; n < batch_size; ++n)
            {
                sum += grad_output[n * output_features + out];
            }

            grad_bias[out] = sum;
        }

    }

    void backward_linear_grad_bias_cuda_float32(
        const Tensor &grad_output,
        Tensor &grad_bias)
    {
        const std::size_t batch_size = grad_output.shape()[0];
        const std::size_t output_features = grad_output.shape()[1];

        const float *grad_output_data =
            static_cast<const float *>(grad_output.data());

        float *grad_bias_data =
            static_cast<float *>(grad_bias.data());

        constexpr int block_size = 256;

        dim3 block(block_size);
        dim3 grid(static_cast<unsigned int>((output_features + block_size - 1) / block_size));

        backward_linear_grad_bias_cuda_float32_kernel<<<grid, block>>>(
            grad_output_data,
            grad_bias_data,
            batch_size,
            output_features);

        check_cuda(
            cudaGetLastError(),
            "CUDA backward linear grad_bias kernel launch failed");
    }

}