#include <kernels/cpu/pooling/backward_avgpool2d_cpu_float32.hpp>

#include <backend/cpu/cpu_parallel.hpp>

#include <cstddef>

namespace kl
{

    namespace
    {

        struct ZeroGradInputTask
        {
            float *grad_input;

            void operator()(std::size_t begin, std::size_t end) const
            {
                for (std::size_t i = begin; i < end; ++i)
                {
                    grad_input[i] = 0.0f;
                }
            }
        };

    }

    void backward_avgpool2d_cpu_float32(
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

        ZeroGradInputTask zero_task{
            grad_input_data};

        cpu_parallel_for(
            0,
            grad_input.numel(),
            zero_task);

        for (std::size_t n = 0; n < batch_size; ++n)
        {
            for (std::size_t c = 0; c < channels; ++c)
            {
                for (std::size_t oh = 0; oh < output_h; ++oh)
                {
                    for (std::size_t ow = 0; ow < output_w; ++ow)
                    {
                        const std::ptrdiff_t h_start =
                            static_cast<std::ptrdiff_t>(oh * options.stride_h) -
                            static_cast<std::ptrdiff_t>(options.padding_h);

                        const std::ptrdiff_t w_start =
                            static_cast<std::ptrdiff_t>(ow * options.stride_w) -
                            static_cast<std::ptrdiff_t>(options.padding_w);

                        const std::ptrdiff_t h_end =
                            h_start + static_cast<std::ptrdiff_t>(options.kernel_h);

                        const std::ptrdiff_t w_end =
                            w_start + static_cast<std::ptrdiff_t>(options.kernel_w);

                        std::size_t valid_count = 0;

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

                                ++valid_count;
                            }
                        }

                        if (valid_count == 0)
                        {
                            continue;
                        }

                        const std::size_t grad_output_index =
                            ((n * channels + c) * output_h + oh) * output_w + ow;

                        const float grad =
                            grad_output_data[grad_output_index] /
                            static_cast<float>(valid_count);

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

                                const std::size_t grad_input_index =
                                    ((n * channels + c) * input_h +
                                     static_cast<std::size_t>(h)) *
                                        input_w +
                                    static_cast<std::size_t>(w);

                                grad_input_data[grad_input_index] += grad;
                            }
                        }
                    }
                }
            }
        }
    }

}