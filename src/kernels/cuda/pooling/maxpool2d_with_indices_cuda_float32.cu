#include <kernels/cuda/pooling/maxpool2d_with_indices_cuda_float32.cuh>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
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

        __global__ void maxpool2d_with_indices_cuda_float32_kernel(
            const float *input,
            float *result,
            std::int32_t *indices,
            std::size_t n_size,
            std::size_t c_size,
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
                n_size * c_size * output_h * output_w;

            if (index >= total)
            {
                return;
            }

            const std::size_t ow = index % output_w;
            const std::size_t oh = (index / output_w) % output_h;
            const std::size_t c = (index / (output_w * output_h)) % c_size;
            const std::size_t n = index / (output_w * output_h * c_size);

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

            float max_value = -INFINITY;
            std::size_t max_index = 0;

            for (long h = h_start; h < h_end; ++h)
            {
                if (h < 0 || h >= static_cast<long>(input_h))
                {
                    continue;
                }

                for (long w = w_start; w < w_end; ++w)
                {
                    if (w < 0 || w >= static_cast<long>(input_w))
                    {
                        continue;
                    }

                    const std::size_t input_index =
                        ((n * c_size + c) * input_h +
                         static_cast<std::size_t>(h)) *
                            input_w +
                        static_cast<std::size_t>(w);

                    const float value = input[input_index];

                    if (value > max_value)
                    {
                        max_value = value;
                        max_index = input_index;
                    }
                }
            }

            result[index] = max_value;
            indices[index] = static_cast<std::int32_t>(max_index);
        }

    }

    void maxpool2d_with_indices_cuda_float32(
        const Tensor &input,
        Tensor &result,
        Tensor &indices,
        const Pooling2dOptions &options)
    {
        const std::size_t n_size = input.shape()[0];
        const std::size_t c_size = input.shape()[1];
        const std::size_t input_h = input.shape()[2];
        const std::size_t input_w = input.shape()[3];

        const std::size_t output_h = result.shape()[2];
        const std::size_t output_w = result.shape()[3];

        const float *input_data =
            static_cast<const float *>(input.data());

        float *result_data =
            static_cast<float *>(result.data());

        std::int32_t *indices_data =
            static_cast<std::int32_t *>(indices.data());

        const std::size_t total =
            n_size * c_size * output_h * output_w;

        constexpr int block_size = 256;

        dim3 block(block_size);
        dim3 grid(static_cast<unsigned int>(
            (total + block_size - 1) / block_size));

        maxpool2d_with_indices_cuda_float32_kernel<<<grid, block>>>(
            input_data,
            result_data,
            indices_data,
            n_size,
            c_size,
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

        check_cuda(
            cudaGetLastError(),
            "CUDA maxpool2d_with_indices kernel launch failed");
    }

}