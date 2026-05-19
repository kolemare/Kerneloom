#include <kernels/cuda/linear/backward_linear_grad_input_cuda_float32.cuh>

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

        __global__ void backward_linear_grad_input_cuda_float32_kernel(
            const float *weights,
            const float *grad_output,
            float *grad_input,
            std::size_t batch_size,
            std::size_t input_features,
            std::size_t output_features)
        {
            const std::size_t index =
                blockIdx.x * blockDim.x + threadIdx.x;

            const std::size_t total = batch_size * input_features;

            if (index >= total)
            {
                return;
            }

            const std::size_t n = index / input_features;
            const std::size_t in = index % input_features;

            float sum = 0.0f;

            for (std::size_t out = 0; out < output_features; ++out)
            {
                sum += grad_output[n * output_features + out] *
                       weights[out * input_features + in];
            }

            grad_input[index] = sum;
        }

    }

    void backward_linear_grad_input_cuda_float32(
        const Tensor &weights,
        const Tensor &grad_output,
        Tensor &grad_input)
    {
        const std::size_t batch_size = grad_output.shape()[0];
        const std::size_t output_features = grad_output.shape()[1];
        const std::size_t input_features = weights.shape()[1];

        const float *weights_data =
            static_cast<const float *>(weights.data());

        const float *grad_output_data =
            static_cast<const float *>(grad_output.data());

        float *grad_input_data =
            static_cast<float *>(grad_input.data());

        const std::size_t total = batch_size * input_features;

        constexpr int block_size = 256;

        dim3 block(block_size);
        dim3 grid(static_cast<unsigned int>((total + block_size - 1) / block_size));

        backward_linear_grad_input_cuda_float32_kernel<<<grid, block>>>(
            weights_data,
            grad_output_data,
            grad_input_data,
            batch_size,
            input_features,
            output_features);

        check_cuda(
            cudaGetLastError(),
            "CUDA backward linear grad_input kernel launch failed");

        check_cuda(
            cudaDeviceSynchronize(),
            "CUDA backward linear grad_input synchronization failed");
    }

}