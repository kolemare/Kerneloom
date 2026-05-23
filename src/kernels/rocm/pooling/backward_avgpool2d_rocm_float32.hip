#include <kernels/rocm/pooling/backward_avgpool2d_rocm_float32.hiph>

#include <hip/hip_runtime.h>

#include <cstddef>
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

        __global__ void zero_float32_kernel(
            float *data,
            std::size_t count)
        {
            const std::size_t index =
                blockIdx.x * blockDim.x + threadIdx.x;

            if (index < count)
            {
                data[index] = 0.0f;
            }
        }

        __global__ void backward_avgpool2d_rocm_float32_kernel(
            const float *grad_output,
            float *grad_input,
            std::size_t batch_size,
            std::size_t channels,
            std::size_t input_h,
            std::size_t input_w,
            std::size_t output_h,
            std::size_t output_w,
            std::size_t kernel_h,
            std::size_t kernel_w,
            std::size_t stride_h,
            std::size_t stride_w,
            std::size_t padding_h,
            std::size_t padding_w)
        {
            const std::size_t index =
                blockIdx.x * blockDim.x + threadIdx.x;

            const std::size_t total =
                batch_size * channels * output_h * output_w;

            if (index >= total)
            {
                return;
            }

            const std::size_t ow = index % output_w;
            const std::size_t oh = (index / output_w) % output_h;
            const std::size_t c = (index / (output_w * output_h)) % channels;
            const std::size_t n = index / (output_w * output_h * channels);

            const long h_start =
                static_cast<long>(oh * stride_h) -
                static_cast<long>(padding_h);

            const long w_start =
                static_cast<long>(ow * stride_w) -
                static_cast<long>(padding_w);

            const long h_end =
                h_start + static_cast<long>(kernel_h);

            const long w_end =
                w_start + static_cast<long>(kernel_w);

            const long h_begin =
                h_start < 0 ? 0 : h_start;

            const long w_begin =
                w_start < 0 ? 0 : w_start;

            const long h_limit =
                h_end > static_cast<long>(input_h)
                    ? static_cast<long>(input_h)
                    : h_end;

            const long w_limit =
                w_end > static_cast<long>(input_w)
                    ? static_cast<long>(input_w)
                    : w_end;

            const std::size_t valid_count =
                static_cast<std::size_t>(h_limit - h_begin) *
                static_cast<std::size_t>(w_limit - w_begin);

            if (valid_count == 0)
            {
                return;
            }

            const float grad =
                grad_output[index] / static_cast<float>(valid_count);

            for (long h = h_begin; h < h_limit; ++h)
            {
                for (long w = w_begin; w < w_limit; ++w)
                {
                    const std::size_t grad_input_index =
                        ((n * channels + c) * input_h +
                         static_cast<std::size_t>(h)) *
                            input_w +
                        static_cast<std::size_t>(w);

                    atomicAdd(
                        &grad_input[grad_input_index],
                        grad);
                }
            }
        }

    }

    void backward_avgpool2d_rocm_float32(
        const Tensor &grad_output,
        Tensor &grad_input,
        const Pooling2dOptions &options)
    {
        const std::size_t batch_size = grad_input.shape()[0];
        const std::size_t channels = grad_input.shape()[1];
        const std::size_t input_h = grad_input.shape()[2];
        const std::size_t input_w = grad_input.shape()[3];

        const std::size_t output_h = grad_output.shape()[2];
        const std::size_t output_w = grad_output.shape()[3];

        const float *grad_output_data =
            static_cast<const float *>(grad_output.data());

        float *grad_input_data =
            static_cast<float *>(grad_input.data());

        constexpr int block_size = 256;

        const std::size_t grad_input_count = grad_input.numel();

        dim3 zero_block(block_size);
        dim3 zero_grid(
            static_cast<unsigned int>(
                (grad_input_count + block_size - 1) / block_size));

        hipLaunchKernelGGL(
            zero_float32_kernel,
            zero_grid,
            zero_block,
            0,
            0,
            grad_input_data,
            grad_input_count);

        check_hip(
            hipGetLastError(),
            "ROCm backward avgpool2d zero kernel launch failed");

        const std::size_t total =
            batch_size * channels * output_h * output_w;

        dim3 block(block_size);
        dim3 grid(
            static_cast<unsigned int>(
                (total + block_size - 1) / block_size));

        hipLaunchKernelGGL(
            backward_avgpool2d_rocm_float32_kernel,
            grid,
            block,
            0,
            0,
            grad_output_data,
            grad_input_data,
            batch_size,
            channels,
            input_h,
            input_w,
            output_h,
            output_w,
            options.kernel_h,
            options.kernel_w,
            options.stride_h,
            options.stride_w,
            options.padding_h,
            options.padding_w);

        check_hip(
            hipGetLastError(),
            "ROCm backward avgpool2d kernel launch failed");

        check_hip(
            hipDeviceSynchronize(),
            "ROCm backward avgpool2d synchronization failed");
    }

}