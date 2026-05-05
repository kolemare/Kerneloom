#include <backend/backend.hpp>

#include <cnn/options/conv2d_options.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/tensor.hpp>

#include <ops/conv2d.hpp>

#include <cstdlib>
#include <exception>
#include <iostream>

int main()
{
    try
    {
        std::size_t batch_size = 32;    // N
        std::size_t input_channels = 3; // C
        std::size_t input_h = 1 << 10;  // H
        std::size_t input_w = 1 << 10;  // W

        std::size_t k_filters = 32; // K
        std::size_t k_channels = input_channels;
        std::size_t k_h = 3; // R
        std::size_t k_w = 3; // S

        kl::Conv2dOptions options;
        options.stride_h = 1;
        options.stride_w = 1;
        options.padding_h = 0;
        options.padding_w = 0;
        options.dilation_h = 1;
        options.dilation_w = 1;
        options.use_bias = false;

        kl::Device target = kl::default_device();

        kl::Tensor input(
            kl::Shape{batch_size, input_channels, input_h, input_w},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::NCHW,
            kl::Storage::RowMajor);

        kl::Tensor kernels(
            kl::Shape{k_filters, k_channels, k_h, k_w},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::NCHW,
            kl::Storage::RowMajor);

        float *input_cpu = static_cast<float *>(input.data());
        float *kernels_cpu = static_cast<float *>(kernels.data());

        for (std::size_t i = 0; i < input.numel(); ++i)
        {
            input_cpu[i] = 1.0f;
        }

        for (std::size_t i = 0; i < kernels.numel(); ++i)
        {
            kernels_cpu[i] = 2.0f;
        }

        kl::Tensor input_target = input.to(target);
        kl::Tensor kernels_target = kernels.to(target);

        const std::size_t effective_kernel_h =
            options.dilation_h * (k_h - 1) + 1;

        const std::size_t effective_kernel_w =
            options.dilation_w * (k_w - 1) + 1;

        const std::size_t output_h =
            (input_h + 2 * options.padding_h - effective_kernel_h) / options.stride_h + 1;

        const std::size_t output_w =
            (input_w + 2 * options.padding_w - effective_kernel_w) / options.stride_w + 1;

        kl::Tensor result_target(
            kl::Shape{batch_size, k_filters, output_h, output_w},
            kl::DType::Float32,
            target,
            kl::Layout::NCHW,
            kl::Storage::RowMajor);

        kl::conv2d(
            input_target,
            kernels_target,
            nullptr,
            result_target,
            options);

        kl::Tensor cpu_result = result_target.to(kl::Device::cpu());

        const float *result = static_cast<const float *>(cpu_result.data());

        std::cout << result[346] << ": "
                  << result[34] << ": "
                  << result[262] << '\n';

        return EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }
}