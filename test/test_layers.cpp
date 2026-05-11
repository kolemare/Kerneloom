#include <backend/backend.hpp>

#include <cnn/layers/activation_layer.hpp>
#include <cnn/layers/conv2d_layer.hpp>
#include <cnn/layers/maxpool2d_layer.hpp>
#include <cnn/options/conv2d_options.hpp>
#include <cnn/options/pooling2d_options.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/tensor.hpp>
#include <core/tensor_pool.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
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

}

int main()
{
    const kl::Device target = kl::default_device();

    const std::size_t batch_size = 8;
    const std::size_t input_channels = 3;
    const std::size_t input_h = 1 << 12;
    const std::size_t input_w = 1 << 12;

    kl::Tensor input_cpu(
        kl::Shape{batch_size, input_channels, input_h, input_w},
        kl::DType::Float32,
        kl::Device::cpu(),
        kl::Layout::NCHW,
        kl::Storage::RowMajor);

    fill_tensor(input_cpu, 1.0f);

    kl::Tensor input = input_cpu.to(target);

    kl::Conv2dOptions conv_options;
    conv_options.stride_h = 1;
    conv_options.stride_w = 1;
    conv_options.padding_h = 1;
    conv_options.padding_w = 1;
    conv_options.dilation_h = 1;
    conv_options.dilation_w = 1;
    conv_options.use_bias = false;

    kl::Pooling2dOptions pool_options;
    pool_options.kernel_h = 2;
    pool_options.kernel_w = 2;
    pool_options.stride_h = 2;
    pool_options.stride_w = 2;
    pool_options.padding_h = 0;
    pool_options.padding_w = 0;

    std::vector<std::unique_ptr<kl::Layer>> layers;

    layers.push_back(std::make_unique<kl::Conv2dLayer>(
        3, 8, 3, 3, kl::DType::Float32, target, conv_options));
    layers.push_back(std::make_unique<kl::ActivationLayer>(
        kl::ActivationType::ReLU));
    layers.push_back(std::make_unique<kl::MaxPool2dLayer>(
        pool_options));

    layers.push_back(std::make_unique<kl::Conv2dLayer>(
        8, 16, 3, 3, kl::DType::Float32, target, conv_options));
    layers.push_back(std::make_unique<kl::ActivationLayer>(
        kl::ActivationType::ReLU));
    layers.push_back(std::make_unique<kl::MaxPool2dLayer>(
        pool_options));

    layers.push_back(std::make_unique<kl::Conv2dLayer>(
        16, 32, 3, 3, kl::DType::Float32, target, conv_options));
    layers.push_back(std::make_unique<kl::ActivationLayer>(
        kl::ActivationType::ReLU));
    layers.push_back(std::make_unique<kl::MaxPool2dLayer>(
        pool_options));

    kl::TensorPool pool;

    for (std::size_t run = 0; run < 10; ++run)
    {
        const auto start = std::chrono::steady_clock::now();

        kl::Tensor *current = &input;

        for (std::unique_ptr<kl::Layer> &layer : layers)
        {
            current = &layer->forward(*current, pool);
        }

        const auto end = std::chrono::steady_clock::now();

        const auto duration_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "run " << run
                  << " | " << duration_ms << " ms"
                  << " | tensors=" << pool.tensor_count()
                  << '\n';

        pool.reset();
    }

    return EXIT_SUCCESS;
}