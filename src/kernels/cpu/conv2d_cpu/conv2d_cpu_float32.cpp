#include <kernels/cpu/conv2d/conv2d_cpu_float32.hpp>

#include <kernels/cpu/conv2d/conv2d_cpu_float32_general.hpp>
#include <kernels/cpu/conv2d/conv2d_cpu_float32_padded.hpp>
#include <kernels/cpu/conv2d/conv2d_cpu_float32_strided.hpp>
#include <kernels/cpu/conv2d/conv2d_cpu_float32_valid.hpp>

namespace kl
{

    void conv2d_cpu_float32(
        const Tensor &input,
        const Tensor &kernels,
        const Tensor *bias,
        Tensor &result,
        const Conv2dOptions &options)
    {
        const bool is_valid =
            options.stride_h == 1 &&
            options.stride_w == 1 &&
            options.padding_h == 0 &&
            options.padding_w == 0 &&
            options.dilation_h == 1 &&
            options.dilation_w == 1;

        const bool is_padded =
            options.stride_h == 1 &&
            options.stride_w == 1 &&
            (options.padding_h > 0 || options.padding_w > 0) &&
            options.dilation_h == 1 &&
            options.dilation_w == 1;

        const bool is_strided =
            (options.stride_h > 1 || options.stride_w > 1) &&
            options.padding_h == 0 &&
            options.padding_w == 0 &&
            options.dilation_h == 1 &&
            options.dilation_w == 1;

        if (is_valid)
        {
            conv2d_cpu_float32_valid(input, kernels, bias, result, options);
            return;
        }

        if (is_padded)
        {
            conv2d_cpu_float32_padded(input, kernels, bias, result, options);
            return;
        }

        if (is_strided)
        {
            conv2d_cpu_float32_strided(input, kernels, bias, result, options);
            return;
        }

        conv2d_cpu_float32_general(input, kernels, bias, result, options);
    }

}