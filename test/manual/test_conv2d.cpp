#include <backend/backend.hpp>

#include <cnn/options/conv2d_options.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/tensor.hpp>

#include <ops/conv2d.hpp>

#include <kernels/cpu/conv2d/conv2d_cpu_float32_general.hpp>

#if defined(KL_ENABLE_CUDA)
#include <kernels/cuda/conv2d/conv2d_cuda_float32_general.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <kernels/rocm/conv2d/conv2d_rocm_float32_general.hiph>
#endif

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

    std::size_t conv2d_output_size(
        std::size_t input_size,
        std::size_t kernel_size,
        std::size_t padding,
        std::size_t stride,
        std::size_t dilation)
    {
        const std::size_t effective_kernel =
            dilation * (kernel_size - 1) + 1;

        return (input_size + 2 * padding - effective_kernel) / stride + 1;
    }

    void fill_tensor(kl::Tensor &tensor, float value)
    {
        float *data = static_cast<float *>(tensor.data());

        for (std::size_t i = 0; i < tensor.numel(); ++i)
        {
            data[i] = value;
        }
    }

    kl::Tensor make_result(
        const kl::Tensor &input,
        const kl::Tensor &kernels,
        const kl::Conv2dOptions &options)
    {
        const std::size_t batch_size = input.shape()[0];
        const std::size_t output_channels = kernels.shape()[0];

        const std::size_t input_h = input.shape()[2];
        const std::size_t input_w = input.shape()[3];

        const std::size_t kernel_h = kernels.shape()[2];
        const std::size_t kernel_w = kernels.shape()[3];

        const std::size_t output_h = conv2d_output_size(
            input_h,
            kernel_h,
            options.padding_h,
            options.stride_h,
            options.dilation_h);

        const std::size_t output_w = conv2d_output_size(
            input_w,
            kernel_w,
            options.padding_w,
            options.stride_w,
            options.dilation_w);

        return kl::Tensor(
            kl::Shape{batch_size, output_channels, output_h, output_w},
            kl::DType::Float32,
            input.device(),
            kl::Layout::NCHW,
            kl::Storage::RowMajor);
    }

    void run_conv2d_general_direct(
        const kl::Tensor &input,
        const kl::Tensor &kernels,
        kl::Tensor &result,
        const kl::Conv2dOptions &options)
    {
        switch (input.device().type())
        {
        case kl::DeviceType::CPU:
            kl::conv2d_cpu_float32_general(
                input,
                kernels,
                nullptr,
                result,
                options);
            return;

        case kl::DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            kl::conv2d_cuda_float32_general(
                input,
                kernels,
                nullptr,
                result,
                options);
            return;
#else
            throw std::runtime_error("CUDA general conv2d requested but CUDA is not enabled");
#endif

        case kl::DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            kl::conv2d_rocm_float32_general(
                input,
                kernels,
                nullptr,
                result,
                options);
            return;
#else
            throw std::runtime_error("ROCm general conv2d requested but ROCm is not enabled");
#endif

        default:
            throw std::runtime_error("unknown device type");
        }
    }

    void run_conv2d_general_case(
        const std::string &name,
        const kl::Tensor &input,
        const kl::Tensor &kernels,
        const kl::Conv2dOptions &options)
    {
        kl::Tensor result = make_result(input, kernels, options);

        const auto start = std::chrono::steady_clock::now();

        run_conv2d_general_direct(
            input,
            kernels,
            result,
            options);

        const auto end = std::chrono::steady_clock::now();

        const auto duration_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        kl::Tensor result_cpu = result.to(kl::Device::cpu());

        const float *result_data = static_cast<const float *>(result_cpu.data());

        std::cout << name
                  << " general"
                  << " | sample=" << result_data[0]
                  << " | " << duration_ms << " ms"
                  << '\n';
    }

    void run_conv2d_specialized_case(
        const std::string &name,
        const kl::Tensor &input,
        const kl::Tensor &kernels,
        const kl::Conv2dOptions &options)
    {
        kl::Tensor result = make_result(input, kernels, options);

        const auto start = std::chrono::steady_clock::now();

        kl::conv2d(
            input,
            kernels,
            nullptr,
            result,
            options);

        const auto end = std::chrono::steady_clock::now();

        const auto duration_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        kl::Tensor result_cpu = result.to(kl::Device::cpu());

        const float *result_data = static_cast<const float *>(result_cpu.data());

        std::cout << name
                  << " specialized"
                  << " | sample=" << result_data[0]
                  << " | " << duration_ms << " ms"
                  << '\n';
    }

    void compare_conv2d_case(
        const std::string &name,
        const kl::Tensor &input,
        const kl::Tensor &kernels,
        const kl::Conv2dOptions &options)
    {
        run_conv2d_general_case(name, input, kernels, options);
        run_conv2d_specialized_case(name, input, kernels, options);
        std::cout << '\n';
    }

}

int main()
{
    try
    {
        const kl::Device target = kl::default_device();

        std::cout << "Target device: "
                  << kl::to_string(target.type())
                  << '\n';

        std::size_t batch_size = 32;
        std::size_t input_channels = 3;
        std::size_t input_h = 1 << 10;
        std::size_t input_w = 1 << 10;

        std::size_t output_channels = 32;
        std::size_t kernel_h = 3;
        std::size_t kernel_w = 3;

        kl::Tensor input_cpu(
            kl::Shape{batch_size, input_channels, input_h, input_w},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::NCHW,
            kl::Storage::RowMajor);

        kl::Tensor kernels_cpu(
            kl::Shape{output_channels, input_channels, kernel_h, kernel_w},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::NCHW,
            kl::Storage::RowMajor);

        fill_tensor(input_cpu, 1.0f);
        fill_tensor(kernels_cpu, 2.0f);

        kl::Tensor input = input_cpu.to(target);
        kl::Tensor kernels = kernels_cpu.to(target);

        kl::Conv2dOptions valid;
        valid.stride_h = 1;
        valid.stride_w = 1;
        valid.padding_h = 0;
        valid.padding_w = 0;
        valid.dilation_h = 1;
        valid.dilation_w = 1;
        valid.use_bias = false;

        kl::Conv2dOptions padded = valid;
        padded.padding_h = 1;
        padded.padding_w = 1;

        kl::Conv2dOptions strided = valid;
        strided.stride_h = 2;
        strided.stride_w = 2;

        kl::Conv2dOptions general = valid;
        general.stride_h = 2;
        general.stride_w = 2;
        general.padding_h = 1;
        general.padding_w = 1;
        general.dilation_h = 2;
        general.dilation_w = 2;

        compare_conv2d_case("valid", input, kernels, valid);
        compare_conv2d_case("padded", input, kernels, padded);
        compare_conv2d_case("strided", input, kernels, strided);
        compare_conv2d_case("general", input, kernels, general);

        return EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }
}