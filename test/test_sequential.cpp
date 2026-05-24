#include <backend/backend.hpp>

#include <cnn/network/initializer.hpp>
#include <cnn/network/sequential.hpp>
#include <cnn/options/conv2d_options.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/tensor.hpp>

#include <ops/activation.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>

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

    const std::size_t batch_size = 64;
    const std::size_t input_channels = 3;
    const std::size_t input_h = 1 << 9;
    const std::size_t input_w = 1 << 9;

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

    cnn.addFlattenLayer();

    cnn.addFullyConnectedLayer(128);
    cnn.addActivationLayer(kl::ActivationType::ReLU);

    cnn.addFullyConnectedLayer(10);
    cnn.addActivationLayer(kl::ActivationType::ReLU);

    if (!cnn.verify())
    {
        std::cerr << "Sequential verification failed\n";
        return EXIT_FAILURE;
    }

    cnn.initializeWeights(kl::InitializerType::KaimingUniform);
    cnn.initializeBiases(kl::InitializerType::Zeros);
    cnn.prepareTraining();

    for (std::size_t run = 0; run < 10; ++run)
    {
        cnn.reset();

        input.reshape_inplace(
            kl::Shape{batch_size, input_channels, input_h, input_w});

        input.set_layout(kl::Layout::NCHW);

        const auto start = std::chrono::steady_clock::now();

        kl::Tensor &output = cnn.forward(input);

        kl::Tensor grad_output_cpu(
            output.shape(),
            output.dtype(),
            kl::Device::cpu(),
            output.layout(),
            output.storage());

        fill_tensor(grad_output_cpu, 1.0f);

        kl::Tensor grad_output = grad_output_cpu.to(target);

        kl::Tensor &grad_input = cnn.backward(grad_output);

        const auto end = std::chrono::steady_clock::now();

        const auto duration_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "run " << run
                  << " | " << duration_ms << " ms"
                  << " | output_shape="
                  << output.shape()[0] << "x" << output.shape()[1]
                  << " | grad_input_shape="
                  << grad_input.shape()[0] << "x"
                  << grad_input.shape()[1] << "x"
                  << grad_input.shape()[2] << "x"
                  << grad_input.shape()[3]
                  << " | tensors=" << cnn.pooled_tensor_count()
                  << '\n';
    }

    return EXIT_SUCCESS;
}