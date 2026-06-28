#include <kernels/cuda/conv2d/backward_conv2d_grad_weights_cuda_float32.cuh>

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

        __global__ void backward_conv2d_grad_weights_cuda_float32_kernel(
            const float *input,
            const float *grad_output,
            float *grad_weights,
            std::size_t batch_size,
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

            const std::size_t kw = index % kernel_w;
            const std::size_t kh = (index / kernel_w) % kernel_h;
            const std::size_t ic = (index / (kernel_w * kernel_h)) % input_channels;
            const std::size_t oc = index / (kernel_w * kernel_h * input_channels);

            float sum = 0.0f;

            for (std::size_t n = 0; n < batch_size; ++n)
            {
                for (std::size_t oh = 0; oh < output_h; ++oh)
                {
                    const long ih =
                        static_cast<long>(oh * stride_h) +
                        static_cast<long>(kh * dilation_h) -
                        static_cast<long>(padding_h);

                    if (ih < 0 || ih >= static_cast<long>(input_h))
                    {
                        continue;
                    }

                    for (std::size_t ow = 0; ow < output_w; ++ow)
                    {
                        const long iw =
                            static_cast<long>(ow * stride_w) +
                            static_cast<long>(kw * dilation_w) -
                            static_cast<long>(padding_w);

                        if (iw < 0 || iw >= static_cast<long>(input_w))
                        {
                            continue;
                        }

                        const std::size_t input_index =
                            ((n * input_channels + ic) * input_h +
                             static_cast<std::size_t>(ih)) *
                                input_w +
                            static_cast<std::size_t>(iw);

                        const std::size_t grad_output_index =
                            ((n * output_channels + oc) * output_h + oh) *
                                output_w +
                            ow;

                        sum += input[input_index] *
                               grad_output[grad_output_index];
                    }
                }
            }

            grad_weights[index] = sum;
        }

    }

    void backward_conv2d_grad_weights_cuda_float32(
        const Tensor &input,
        const Tensor &grad_output,
        Tensor &grad_weights,
        const Conv2dOptions &options)
    {
        const std::size_t batch_size = input.shape()[0];
        const std::size_t input_channels = input.shape()[1];
        const std::size_t input_h = input.shape()[2];
        const std::size_t input_w = input.shape()[3];

        const std::size_t output_channels = grad_weights.shape()[0];
        const std::size_t kernel_h = grad_weights.shape()[2];
        const std::size_t kernel_w = grad_weights.shape()[3];

        const std::size_t output_h = grad_output.shape()[2];
        const std::size_t output_w = grad_output.shape()[3];

        const float *input_data =
            static_cast<const float *>(input.data());

        const float *grad_output_data =
            static_cast<const float *>(grad_output.data());

        float *grad_weights_data =
            static_cast<float *>(grad_weights.data());

        const std::size_t count = grad_weights.numel();

        constexpr int block_size = 256;

        dim3 block(block_size);
        dim3 grid(static_cast<unsigned int>(
            (count + block_size - 1) / block_size));

        backward_conv2d_grad_weights_cuda_float32_kernel<<<grid, block>>>(
            input_data,
            grad_output_data,
            grad_weights_data,
            batch_size,
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
            "CUDA conv2d backward grad_weights kernel launch failed");
    }

}