#include <kernels/cuda/conv2d/backward_conv2d_grad_input_cuda_float32.cuh>

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

        __global__ void backward_conv2d_grad_input_cuda_float32_kernel(
            const float *weights,
            const float *grad_output,
            float *grad_input,
            std::size_t input_channels,
            std::size_t input_h,
            std::size_t input_w,
            std::size_t output_channels,
            std::size_t output_h,
            std::size_t output_w,
            std::size_t kernel_h,
            std::size_t kernel_w,
            std::size_t stride_h,
            std::size_t stride_w,
            std::size_t padding_h,
            std::size_t padding_w,
            std::size_t dilation_h,
            std::size_t dilation_w,
            std::size_t count)
        {
            const std::size_t index =
                blockIdx.x * blockDim.x + threadIdx.x;

            if (index >= count)
            {
                return;
            }

            const std::size_t iw = index % input_w;
            const std::size_t ih = (index / input_w) % input_h;
            const std::size_t ic = (index / (input_w * input_h)) % input_channels;
            const std::size_t n = index / (input_w * input_h * input_channels);

            float sum = 0.0f;

            for (std::size_t oc = 0; oc < output_channels; ++oc)
            {
                for (std::size_t kh = 0; kh < kernel_h; ++kh)
                {
                    const long numerator_h =
                        static_cast<long>(ih) +
                        static_cast<long>(padding_h) -
                        static_cast<long>(kh * dilation_h);

                    if (numerator_h < 0)
                    {
                        continue;
                    }

                    if (numerator_h % static_cast<long>(stride_h) != 0)
                    {
                        continue;
                    }

                    const long oh =
                        numerator_h / static_cast<long>(stride_h);

                    if (oh < 0 || oh >= static_cast<long>(output_h))
                    {
                        continue;
                    }

                    for (std::size_t kw = 0; kw < kernel_w; ++kw)
                    {
                        const long numerator_w =
                            static_cast<long>(iw) +
                            static_cast<long>(padding_w) -
                            static_cast<long>(kw * dilation_w);

                        if (numerator_w < 0)
                        {
                            continue;
                        }

                        if (numerator_w % static_cast<long>(stride_w) != 0)
                        {
                            continue;
                        }

                        const long ow =
                            numerator_w / static_cast<long>(stride_w);

                        if (ow < 0 || ow >= static_cast<long>(output_w))
                        {
                            continue;
                        }

                        const std::size_t grad_output_index =
                            ((n * output_channels + oc) * output_h +
                             static_cast<std::size_t>(oh)) *
                                output_w +
                            static_cast<std::size_t>(ow);

                        const std::size_t weight_index =
                            ((oc * input_channels + ic) * kernel_h + kh) *
                                kernel_w +
                            kw;

                        sum += grad_output[grad_output_index] *
                               weights[weight_index];
                    }
                }
            }

            grad_input[index] = sum;
        }

    }

    void backward_conv2d_grad_input_cuda_float32(
        const Tensor &weights,
        const Tensor &grad_output,
        Tensor &grad_input,
        const Conv2dOptions &options)
    {
        const std::size_t input_channels = grad_input.shape()[1];
        const std::size_t input_h = grad_input.shape()[2];
        const std::size_t input_w = grad_input.shape()[3];

        const std::size_t output_channels = weights.shape()[0];
        const std::size_t kernel_h = weights.shape()[2];
        const std::size_t kernel_w = weights.shape()[3];

        const std::size_t output_h = grad_output.shape()[2];
        const std::size_t output_w = grad_output.shape()[3];

        const float *weights_data =
            static_cast<const float *>(weights.data());

        const float *grad_output_data =
            static_cast<const float *>(grad_output.data());

        float *grad_input_data =
            static_cast<float *>(grad_input.data());

        const std::size_t count = grad_input.numel();

        constexpr int block_size = 256;

        dim3 block(block_size);
        dim3 grid(static_cast<unsigned int>(
            (count + block_size - 1) / block_size));

        backward_conv2d_grad_input_cuda_float32_kernel<<<grid, block>>>(
            weights_data,
            grad_output_data,
            grad_input_data,
            input_channels,
            input_h,
            input_w,
            output_channels,
            output_h,
            output_w,
            kernel_h,
            kernel_w,
            options.stride_h,
            options.stride_w,
            options.padding_h,
            options.padding_w,
            options.dilation_h,
            options.dilation_w,
            count);

        check_cuda(
            cudaGetLastError(),
            "CUDA conv2d backward grad_input kernel launch failed");

        check_cuda(
            cudaDeviceSynchronize(),
            "CUDA conv2d backward grad_input synchronization failed");
    }

}