#include <backend/backend.hpp>

#include <cnn/options/pooling2d_options.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/tensor.hpp>

#include <ops/avgpool2d.hpp>
#include <ops/maxpool2d.hpp>

#include <cstdlib>
#include <exception>
#include <iostream>

namespace
{

    std::size_t pooling2d_output_size(
        std::size_t input_size,
        std::size_t kernel_size,
        std::size_t padding,
        std::size_t stride)
    {
        return (input_size + 2 * padding - kernel_size) / stride + 1;
    }

    void fill_input(kl::Tensor &tensor)
    {
        float *data = static_cast<float *>(tensor.data());

        for (std::size_t i = 0; i < tensor.numel(); ++i)
        {
            data[i] = static_cast<float>(i + 1);
        }
    }

    void print_tensor_4d_nchw(
        const kl::Tensor &tensor)
    {
        const float *data = static_cast<const float *>(tensor.data());

        const std::size_t N = tensor.shape()[0];
        const std::size_t C = tensor.shape()[1];
        const std::size_t H = tensor.shape()[2];
        const std::size_t W = tensor.shape()[3];

        for (std::size_t n = 0; n < N; ++n)
        {
            for (std::size_t c = 0; c < C; ++c)
            {
                std::cout << "n=" << n << ", c=" << c << '\n';

                for (std::size_t h = 0; h < H; ++h)
                {
                    for (std::size_t w = 0; w < W; ++w)
                    {
                        const std::size_t index =
                            n * C * H * W +
                            c * H * W +
                            h * W +
                            w;

                        std::cout << data[index] << ' ';
                    }

                    std::cout << '\n';
                }

                std::cout << '\n';
            }
        }
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

        const std::size_t batch_size = 1;
        const std::size_t channels = 1;
        const std::size_t input_h = 4;
        const std::size_t input_w = 4;

        kl::Pooling2dOptions options;
        options.kernel_h = 2;
        options.kernel_w = 2;
        options.stride_h = 2;
        options.stride_w = 2;
        options.padding_h = 0;
        options.padding_w = 0;

        kl::Tensor input_cpu(
            kl::Shape{batch_size, channels, input_h, input_w},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::NCHW,
            kl::Storage::RowMajor);

        fill_input(input_cpu);

        std::cout << "Input:\n";
        print_tensor_4d_nchw(input_cpu);

        kl::Tensor input = input_cpu.to(target);

        const std::size_t output_h = pooling2d_output_size(
            input_h,
            options.kernel_h,
            options.padding_h,
            options.stride_h);

        const std::size_t output_w = pooling2d_output_size(
            input_w,
            options.kernel_w,
            options.padding_w,
            options.stride_w);

        kl::Tensor max_result(
            kl::Shape{batch_size, channels, output_h, output_w},
            kl::DType::Float32,
            target,
            kl::Layout::NCHW,
            kl::Storage::RowMajor);

        kl::Tensor avg_result(
            kl::Shape{batch_size, channels, output_h, output_w},
            kl::DType::Float32,
            target,
            kl::Layout::NCHW,
            kl::Storage::RowMajor);

        kl::maxpool2d(input, max_result, options);
        kl::avgpool2d(input, avg_result, options);

        kl::Tensor max_result_cpu = max_result.to(kl::Device::cpu());
        kl::Tensor avg_result_cpu = avg_result.to(kl::Device::cpu());

        std::cout << "MaxPool2D result:\n";
        print_tensor_4d_nchw(max_result_cpu);

        std::cout << "AvgPool2D result:\n";
        print_tensor_4d_nchw(avg_result_cpu);

        return EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }
}