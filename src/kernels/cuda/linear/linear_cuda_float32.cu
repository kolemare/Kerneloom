#include <kernels/cuda/linear/linear_cuda_float32.cuh>

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

        __global__ void linear_cuda_float32_kernel(
            const float *input,
            const float *weights,
            const float *bias,
            float *result,
            std::size_t batch_size,
            std::size_t input_features,
            std::size_t output_features)
        {
            const std::size_t index =
                blockIdx.x * blockDim.x + threadIdx.x;

            const std::size_t total = batch_size * output_features;

            if (index >= total)
            {
                return;
            }

            const std::size_t n = index / output_features;
            const std::size_t out = index % output_features;

            const std::size_t input_offset = n * input_features;
            const std::size_t weight_offset = out * input_features;

            float sum = 0.0f;

            for (std::size_t in = 0; in < input_features; ++in)
            {
                sum += input[input_offset + in] *
                       weights[weight_offset + in];
            }

            if (bias != nullptr)
            {
                sum += bias[out];
            }

            result[index] = sum;
        }

    }

    void linear_cuda_float32(
        const Tensor &input,
        const Tensor &weights,
        const Tensor *bias,
        Tensor &result)
    {
        const std::size_t batch_size = input.shape()[0];
        const std::size_t input_features = input.shape()[1];
        const std::size_t output_features = weights.shape()[0];

        const float *input_data = static_cast<const float *>(input.data());
        const float *weights_data = static_cast<const float *>(weights.data());

        const float *bias_data = nullptr;

        if (bias != nullptr)
        {
            bias_data = static_cast<const float *>(bias->data());
        }

        float *result_data = static_cast<float *>(result.data());

        const std::size_t total = batch_size * output_features;

        constexpr int block_size = 256;

        dim3 block(block_size);
        dim3 grid(static_cast<unsigned int>((total + block_size - 1) / block_size));

        linear_cuda_float32_kernel<<<grid, block>>>(
            input_data,
            weights_data,
            bias_data,
            result_data,
            batch_size,
            input_features,
            output_features);

        check_cuda(cudaGetLastError(), "CUDA linear kernel launch failed");
        check_cuda(cudaDeviceSynchronize(), "CUDA linear synchronization failed");
    }

}