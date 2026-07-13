#include <backend/backend.hpp>

#include <cnn/network/sequential.hpp>
#include <cnn/options/conv2d_options.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>

#include <ops/activation.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace
{

    void fill_tensor(kl::Tensor &tensor, float value)
    {
        float *data = static_cast<float *>(tensor.data());

        for (std::size_t i = 0; i < tensor.numel(); ++i)
        {
            data[i] = value;
        }
    }

    std::string shape_to_string(const kl::Shape &shape)
    {
        std::ostringstream out;

        for (std::size_t i = 0; i < shape.rank(); ++i)
        {
            if (i > 0)
            {
                out << 'x';
            }

            out << shape[i];
        }

        return out.str();
    }

    void reset_input_shape(
        kl::Tensor &input,
        std::size_t batch_size,
        std::size_t input_channels,
        std::size_t input_h,
        std::size_t input_w)
    {
        input.reshape_inplace(
            kl::Shape{batch_size, input_channels, input_h, input_w});

        input.set_layout(kl::Layout::NCHW);
    }

} // namespace

int main()
{
    const kl::Device target = kl::default_device();

    const std::size_t batch_size = 8;
    const std::size_t input_channels = 3;
    const std::size_t input_h = 1 << 11;
    const std::size_t input_w = 1 << 11;

    const std::size_t warmup_runs = 3;
    const std::size_t measured_runs = 10;

    std::cout << "Kerneloom CNN forward benchmark\n";
    std::cout << "target_device=" << kl::to_string(target.type()) << '\n';
    std::cout << "input_shape="
              << batch_size << 'x'
              << input_channels << 'x'
              << input_h << 'x'
              << input_w << '\n';
    std::cout << "warmup_runs=" << warmup_runs << '\n';
    std::cout << "measured_runs=" << measured_runs << '\n';
    std::cout << '\n';

    kl::Tensor input_cpu(
        kl::Shape{batch_size, input_channels, input_h, input_w},
        kl::DType::Float32,
        kl::Device::cpu(),
        kl::Layout::NCHW,
        kl::Storage::RowMajor);

    fill_tensor(input_cpu, 1.0f);

    kl::Tensor input = input_cpu.to(target);

    kl::Sequential cnn(
        batch_size,
        input_channels,
        input_h,
        input_w,
        kl::DType::Float32,
        target);

    kl::Conv2dOptions conv_options;
    conv_options.stride_h = 1;
    conv_options.stride_w = 1;
    conv_options.padding_h = 1;
    conv_options.padding_w = 1;
    conv_options.dilation_h = 1;
    conv_options.dilation_w = 1;
    conv_options.use_bias = true;

    cnn.addConvolutionLayer(32, 3, 3, conv_options);
    cnn.addActivationLayer(kl::ActivationType::ReLU);
    cnn.addMaxPoolingLayer(2);

    cnn.addConvolutionLayer(64, 3, 3, conv_options);
    cnn.addActivationLayer(kl::ActivationType::ReLU);
    cnn.addMaxPoolingLayer(2);

    cnn.addConvolutionLayer(64, 3, 3, conv_options);
    cnn.addActivationLayer(kl::ActivationType::ReLU);
    cnn.addMaxPoolingLayer(2);

    std::cout << "warming up...\n";

    for (std::size_t run = 0; run < warmup_runs; ++run)
    {
        cnn.reset();

        reset_input_shape(
            input,
            batch_size,
            input_channels,
            input_h,
            input_w);

        kl::Tensor &output = cnn.forward(input);

        std::cout << "warmup " << run
                  << " | output_shape=" << shape_to_string(output.shape())
                  << " | tensors=" << cnn.pooled_tensor_count()
                  << '\n';
    }

    std::cout << '\n';
    std::cout << "measuring...\n";

    std::vector<double> timings_ms;
    timings_ms.reserve(measured_runs);

    std::string last_output_shape;

    for (std::size_t run = 0; run < measured_runs; ++run)
    {
        cnn.reset();

        reset_input_shape(
            input,
            batch_size,
            input_channels,
            input_h,
            input_w);

        const auto start = std::chrono::steady_clock::now();

        kl::Tensor &output = cnn.forward(input);

        const auto end = std::chrono::steady_clock::now();

        last_output_shape = shape_to_string(output.shape());

        const double duration_ms =
            std::chrono::duration<double, std::milli>(end - start).count();

        timings_ms.push_back(duration_ms);

        std::cout << "run " << run
                  << " | " << std::fixed << std::setprecision(3)
                  << duration_ms << " ms"
                  << " | output_shape=" << last_output_shape
                  << " | tensors=" << cnn.pooled_tensor_count()
                  << '\n';
    }

    const double min_ms =
        *std::min_element(timings_ms.begin(), timings_ms.end());

    const double max_ms =
        *std::max_element(timings_ms.begin(), timings_ms.end());

    const double avg_ms =
        std::accumulate(timings_ms.begin(), timings_ms.end(), 0.0) /
        static_cast<double>(timings_ms.size());

    std::cout << '\n';
    std::cout << "summary\n";
    std::cout << "target_device=" << kl::to_string(target.type()) << '\n';
    std::cout << "input_shape="
              << batch_size << 'x'
              << input_channels << 'x'
              << input_h << 'x'
              << input_w << '\n';
    std::cout << "output_shape=" << last_output_shape << '\n';
    std::cout << "min_ms=" << std::fixed << std::setprecision(3) << min_ms << '\n';
    std::cout << "avg_ms=" << std::fixed << std::setprecision(3) << avg_ms << '\n';
    std::cout << "max_ms=" << std::fixed << std::setprecision(3) << max_ms << '\n';

    return EXIT_SUCCESS;
}