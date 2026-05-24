#include <kernels/cuda/conv2d/backward_conv2d_grad_bias_cuda_float32.cuh>

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

        __global__ void backward_conv2d_grad_bias_cuda_float32_kernel(
            const float *grad_output,
            float *grad_bias,
            std::size_t batch_size,
            std::size_t output_channels,
            std::size_t output_h,
            std::size_t output_w)
        {
            const std::size_t oc =
                blockIdx.x * blockDim.x + threadIdx.x;

            if (oc >= output_channels)
            {
                return;
            }

            float sum = 0.0f;

            for (std::size_t n = 0; n < batch_size; ++n)
            {
                for (std::size_t oh = 0; oh < output_h; ++oh)
                {
                    for (std::size_t ow = 0; ow < output_w; ++ow)
                    {
                        const std::size_t index =
                            ((n * output_channels + oc) * output_h + oh) *
                                output_w +
                            ow;

                        sum += grad_output[index];
                    }
                }
            }

            grad_bias[oc] = sum;
        }

    }

    void backward_conv2d_grad_bias_cuda_float32(
        const Tensor &grad_output,
        Tensor &grad_bias)
    {
        const std::size_t batch_size = grad_output.shape()[0];
        const std::size_t output_channels = grad_output.shape()[1];
        const std::size_t output_h = grad_output.shape()[2];
        const std::size_t output_w = grad_output.shape()[3];

        const float *grad_output_data =
            static_cast<const float *>(grad_output.data());

        float *grad_bias_data =
            static_cast<float *>(grad_bias.data());

        constexpr int block_size = 256;

        dim3 block(block_size);
        dim3 grid(static_cast<unsigned int>(
            (output_channels + block_size - 1) / block_size));

        backward_conv2d_grad_bias_cuda_float32_kernel<<<grid, block>>>(
            grad_output_data,
            grad_bias_data,
            batch_size,
            output_channels,
            output_h,
            output_w);

        check_cuda(
            cudaGetLastError(),
            "CUDA conv2d backward grad_bias kernel launch failed");

        check_cuda(
            cudaDeviceSynchronize(),
            "CUDA conv2d backward grad_bias synchronization failed");
    }

}