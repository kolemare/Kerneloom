#include <kernels/cuda/linear/backward_linear_grad_weights_cuda_float32.cuh>

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

        __global__ void backward_linear_grad_weights_cuda_float32_kernel(
            const float *input,
            const float *grad_output,
            float *grad_weights,
            std::size_t batch_size,
            std::size_t input_features,
            std::size_t output_features)
        {
            const std::size_t index =
                blockIdx.x * blockDim.x + threadIdx.x;

            const std::size_t total = output_features * input_features;

            if (index >= total)
            {
                return;
            }

            const std::size_t out = index / input_features;
            const std::size_t in = index % input_features;

            float sum = 0.0f;

            for (std::size_t n = 0; n < batch_size; ++n)
            {
                sum += grad_output[n * output_features + out] *
                       input[n * input_features + in];
            }

            grad_weights[index] = sum;
        }

    }

    void backward_linear_grad_weights_cuda_float32(
        const Tensor &input,
        const Tensor &grad_output,
        Tensor &grad_weights)
    {
        const std::size_t batch_size = input.shape()[0];
        const std::size_t input_features = input.shape()[1];
        const std::size_t output_features = grad_output.shape()[1];

        const float *input_data =
            static_cast<const float *>(input.data());

        const float *grad_output_data =
            static_cast<const float *>(grad_output.data());

        float *grad_weights_data =
            static_cast<float *>(grad_weights.data());

        const std::size_t total = output_features * input_features;

        constexpr int block_size = 256;

        dim3 block(block_size);
        dim3 grid(static_cast<unsigned int>((total + block_size - 1) / block_size));

        backward_linear_grad_weights_cuda_float32_kernel<<<grid, block>>>(
            input_data,
            grad_output_data,
            grad_weights_data,
            batch_size,
            input_features,
            output_features);

        check_cuda(
            cudaGetLastError(),
            "CUDA backward linear grad_weights kernel launch failed");

        check_cuda(
            cudaDeviceSynchronize(),
            "CUDA backward linear grad_weights synchronization failed");
    }

}