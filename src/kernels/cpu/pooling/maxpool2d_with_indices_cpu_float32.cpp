#include <kernels/cpu/pooling/maxpool2d_with_indices_cpu_float32.hpp>

#include <backend/cpu/cpu_parallel.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace kl
{

    namespace
    {

        struct MaxPool2dWithIndicesCpuFloat32Task
        {
            const float *input;
            float *result;
            std::int32_t *indices;

            std::size_t n_size;
            std::size_t c_size;
            std::size_t input_h;
            std::size_t input_w;
            std::size_t output_h;
            std::size_t output_w;

            std::size_t kernel_h;
            std::size_t kernel_w;
            std::size_t stride_h;
            std::size_t stride_w;
            std::size_t padding_h;
            std::size_t padding_w;

            void operator()(std::size_t begin, std::size_t end) const
            {
                (void)n_size;

                for (std::size_t index = begin; index < end; ++index)
                {
                    const std::size_t ow = index % output_w;
                    const std::size_t oh = (index / output_w) % output_h;
                    const std::size_t c = (index / (output_w * output_h)) % c_size;
                    const std::size_t n = index / (output_w * output_h * c_size);

                    const std::ptrdiff_t h_start =
                        static_cast<std::ptrdiff_t>(oh * stride_h) -
                        static_cast<std::ptrdiff_t>(padding_h);

                    const std::ptrdiff_t w_start =
                        static_cast<std::ptrdiff_t>(ow * stride_w) -
                        static_cast<std::ptrdiff_t>(padding_w);

                    const std::ptrdiff_t h_end =
                        h_start + static_cast<std::ptrdiff_t>(kernel_h);

                    const std::ptrdiff_t w_end =
                        w_start + static_cast<std::ptrdiff_t>(kernel_w);

                    float max_value = -std::numeric_limits<float>::infinity();
                    std::size_t max_index = 0;

                    for (std::ptrdiff_t h = h_start; h < h_end; ++h)
                    {
                        if (h < 0 || h >= static_cast<std::ptrdiff_t>(input_h))
                        {
                            continue;
                        }

                        for (std::ptrdiff_t w = w_start; w < w_end; ++w)
                        {
                            if (w < 0 || w >= static_cast<std::ptrdiff_t>(input_w))
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
        };

    }

    void maxpool2d_with_indices_cpu_float32(
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

        MaxPool2dWithIndicesCpuFloat32Task task{
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
            options.padding_w};

        cpu_parallel_for(
            0,
            n_size * c_size * output_h * output_w,
            task);
    }

}