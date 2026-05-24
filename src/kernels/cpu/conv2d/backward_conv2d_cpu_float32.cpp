#include <kernels/cpu/conv2d/backward_conv2d_cpu_float32.hpp>

#include <backend/cpu/cpu_parallel.hpp>

#include <cstddef>

namespace kl
{

    namespace
    {

        struct GradInputTask
        {
            const float *weights;
            const float *grad_output;
            float *grad_input;

            std::size_t input_channels;
            std::size_t input_h;
            std::size_t input_w;

            std::size_t output_channels;
            std::size_t output_h;
            std::size_t output_w;

            std::size_t kernel_h;
            std::size_t kernel_w;

            std::size_t stride_h;
            std::size_t stride_w;
            std::size_t padding_h;
            std::size_t padding_w;
            std::size_t dilation_h;
            std::size_t dilation_w;

            void operator()(std::size_t begin, std::size_t end) const
            {
                for (std::size_t index = begin; index < end; ++index)
                {
                    const std::size_t iw = index % input_w;
                    const std::size_t ih = (index / input_w) % input_h;
                    const std::size_t ic = (index / (input_w * input_h)) % input_channels;
                    const std::size_t n = index / (input_w * input_h * input_channels);

                    float sum = 0.0f;

                    for (std::size_t oc = 0; oc < output_channels; ++oc)
                    {
                        for (std::size_t kh = 0; kh < kernel_h; ++kh)
                        {
                            const std::ptrdiff_t numerator_h =
                                static_cast<std::ptrdiff_t>(ih) +
                                static_cast<std::ptrdiff_t>(padding_h) -
                                static_cast<std::ptrdiff_t>(kh * dilation_h);

                            if (numerator_h < 0)
                            {
                                continue;
                            }

                            if (numerator_h % static_cast<std::ptrdiff_t>(stride_h) != 0)
                            {
                                continue;
                            }

                            const std::ptrdiff_t oh =
                                numerator_h / static_cast<std::ptrdiff_t>(stride_h);

                            if (oh < 0 || oh >= static_cast<std::ptrdiff_t>(output_h))
                            {
                                continue;
                            }

                            for (std::size_t kw = 0; kw < kernel_w; ++kw)
                            {
                                const std::ptrdiff_t numerator_w =
                                    static_cast<std::ptrdiff_t>(iw) +
                                    static_cast<std::ptrdiff_t>(padding_w) -
                                    static_cast<std::ptrdiff_t>(kw * dilation_w);

                                if (numerator_w < 0)
                                {
                                    continue;
                                }

                                if (numerator_w % static_cast<std::ptrdiff_t>(stride_w) != 0)
                                {
                                    continue;
                                }

                                const std::ptrdiff_t ow =
                                    numerator_w / static_cast<std::ptrdiff_t>(stride_w);

                                if (ow < 0 || ow >= static_cast<std::ptrdiff_t>(output_w))
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
        };

        struct GradWeightsTask
        {
            const float *input;
            const float *grad_output;
            float *grad_weights;

            std::size_t batch_size;
            std::size_t input_channels;
            std::size_t input_h;
            std::size_t input_w;

            std::size_t output_channels;
            std::size_t output_h;
            std::size_t output_w;

            std::size_t kernel_h;
            std::size_t kernel_w;

            std::size_t stride_h;
            std::size_t stride_w;
            std::size_t padding_h;
            std::size_t padding_w;
            std::size_t dilation_h;
            std::size_t dilation_w;

            void operator()(std::size_t begin, std::size_t end) const
            {
                for (std::size_t index = begin; index < end; ++index)
                {
                    const std::size_t kw = index % kernel_w;
                    const std::size_t kh = (index / kernel_w) % kernel_h;
                    const std::size_t ic = (index / (kernel_w * kernel_h)) % input_channels;
                    const std::size_t oc = index / (kernel_w * kernel_h * input_channels);

                    float sum = 0.0f;

                    for (std::size_t n = 0; n < batch_size; ++n)
                    {
                        for (std::size_t oh = 0; oh < output_h; ++oh)
                        {
                            const std::ptrdiff_t ih =
                                static_cast<std::ptrdiff_t>(oh * stride_h) +
                                static_cast<std::ptrdiff_t>(kh * dilation_h) -
                                static_cast<std::ptrdiff_t>(padding_h);

                            if (ih < 0 || ih >= static_cast<std::ptrdiff_t>(input_h))
                            {
                                continue;
                            }

                            for (std::size_t ow = 0; ow < output_w; ++ow)
                            {
                                const std::ptrdiff_t iw =
                                    static_cast<std::ptrdiff_t>(ow * stride_w) +
                                    static_cast<std::ptrdiff_t>(kw * dilation_w) -
                                    static_cast<std::ptrdiff_t>(padding_w);

                                if (iw < 0 || iw >= static_cast<std::ptrdiff_t>(input_w))
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
        };

        struct GradBiasTask
        {
            const float *grad_output;
            float *grad_bias;

            std::size_t batch_size;
            std::size_t output_channels;
            std::size_t output_h;
            std::size_t output_w;

            void operator()(std::size_t begin, std::size_t end) const
            {
                for (std::size_t oc = begin; oc < end; ++oc)
                {
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
        };

    }

    void backward_conv2d_cpu_float32(
        const Tensor &input,
        const Tensor &weights,
        const Tensor &grad_output,
        Tensor &grad_input,
        Tensor &grad_weights,
        Tensor *grad_bias,
        const Conv2dOptions &options)
    {
        const std::size_t batch_size = input.shape()[0];
        const std::size_t input_channels = input.shape()[1];
        const std::size_t input_h = input.shape()[2];
        const std::size_t input_w = input.shape()[3];

        const std::size_t output_channels = weights.shape()[0];
        const std::size_t kernel_h = weights.shape()[2];
        const std::size_t kernel_w = weights.shape()[3];

        const std::size_t output_h = grad_output.shape()[2];
        const std::size_t output_w = grad_output.shape()[3];

        const float *input_data =
            static_cast<const float *>(input.data());

        const float *weights_data =
            static_cast<const float *>(weights.data());

        const float *grad_output_data =
            static_cast<const float *>(grad_output.data());

        float *grad_input_data =
            static_cast<float *>(grad_input.data());

        float *grad_weights_data =
            static_cast<float *>(grad_weights.data());

        GradInputTask grad_input_task{
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
            options.dilation_w};

        cpu_parallel_for(
            0,
            grad_input.numel(),
            grad_input_task);

        GradWeightsTask grad_weights_task{
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
            options.dilation_w};

        cpu_parallel_for(
            0,
            grad_weights.numel(),
            grad_weights_task);

        if (grad_bias != nullptr)
        {
            float *grad_bias_data =
                static_cast<float *>(grad_bias->data());

            GradBiasTask grad_bias_task{
                grad_output_data,
                grad_bias_data,
                batch_size,
                output_channels,
                output_h,
                output_w};

            cpu_parallel_for(
                0,
                output_channels,
                grad_bias_task);
        }
    }

}