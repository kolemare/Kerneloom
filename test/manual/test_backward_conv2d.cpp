#include <backend/backend.hpp>

#include <cnn/options/conv2d_options.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/tensor.hpp>

#include <ops/backward_conv2d.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace
{

    void fill_tensor(
        kl::Tensor &tensor,
        const float *values)
    {
        float *data = static_cast<float *>(tensor.data());

        for (std::size_t i = 0; i < tensor.numel(); ++i)
        {
            data[i] = values[i];
        }
    }

    bool close_enough(
        float actual,
        float expected)
    {
        return std::fabs(actual - expected) < 1.0e-5f;
    }

    bool check_tensor(
        const kl::Tensor &tensor,
        const float *expected)
    {
        const float *data = static_cast<const float *>(tensor.data());

        for (std::size_t i = 0; i < tensor.numel(); ++i)
        {
            if (!close_enough(data[i], expected[i]))
            {
                std::cout << "mismatch at " << i
                          << " | actual=" << data[i]
                          << " | expected=" << expected[i]
                          << '\n';

                return false;
            }
        }

        return true;
    }

}

int main()
{
    const kl::Device target = kl::default_device();

    kl::Conv2dOptions options;
    options.stride_h = 1;
    options.stride_w = 1;
    options.padding_h = 0;
    options.padding_w = 0;
    options.dilation_h = 1;
    options.dilation_w = 1;
    options.use_bias = true;

    kl::Tensor input_cpu(
        kl::Shape{1, 1, 3, 3},
        kl::DType::Float32,
        kl::Device::cpu(),
        kl::Layout::NCHW,
        kl::Storage::RowMajor);

    kl::Tensor weights_cpu(
        kl::Shape{1, 1, 2, 2},
        kl::DType::Float32,
        kl::Device::cpu(),
        kl::Layout::NCHW,
        kl::Storage::RowMajor);

    kl::Tensor grad_output_cpu(
        kl::Shape{1, 1, 2, 2},
        kl::DType::Float32,
        kl::Device::cpu(),
        kl::Layout::NCHW,
        kl::Storage::RowMajor);

    const float input_values[9] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f};

    const float weight_values[4] = {
        1.0f, 2.0f,
        3.0f, 4.0f};

    const float grad_output_values[4] = {
        1.0f, 1.0f,
        1.0f, 1.0f};

    fill_tensor(input_cpu, input_values);
    fill_tensor(weights_cpu, weight_values);
    fill_tensor(grad_output_cpu, grad_output_values);

    kl::Tensor input = input_cpu.to(target);
    kl::Tensor weights = weights_cpu.to(target);
    kl::Tensor grad_output = grad_output_cpu.to(target);

    kl::Tensor grad_input(
        input.shape(),
        kl::DType::Float32,
        target,
        kl::Layout::NCHW,
        kl::Storage::RowMajor);

    kl::Tensor grad_weights(
        weights.shape(),
        kl::DType::Float32,
        target,
        kl::Layout::NCHW,
        kl::Storage::RowMajor);

    kl::Tensor grad_bias(
        kl::Shape{1},
        kl::DType::Float32,
        target,
        kl::Layout::Unknown,
        kl::Storage::RowMajor);

    kl::backward_conv2d(
        input,
        weights,
        grad_output,
        grad_input,
        grad_weights,
        &grad_bias,
        options);

    kl::Tensor grad_input_cpu = grad_input.to(kl::Device::cpu());
    kl::Tensor grad_weights_cpu = grad_weights.to(kl::Device::cpu());
    kl::Tensor grad_bias_cpu = grad_bias.to(kl::Device::cpu());

    const float expected_grad_input[9] = {
        1.0f, 3.0f, 2.0f,
        4.0f, 10.0f, 6.0f,
        3.0f, 7.0f, 4.0f};

    const float expected_grad_weights[4] = {
        12.0f, 16.0f,
        24.0f, 28.0f};

    const float expected_grad_bias[1] = {
        4.0f};

    bool passed = true;

    if (!check_tensor(grad_input_cpu, expected_grad_input))
    {
        std::cout << "grad_input failed\n";
        passed = false;
    }

    if (!check_tensor(grad_weights_cpu, expected_grad_weights))
    {
        std::cout << "grad_weights failed\n";
        passed = false;
    }

    if (!check_tensor(grad_bias_cpu, expected_grad_bias))
    {
        std::cout << "grad_bias failed\n";
        passed = false;
    }

    if (!passed)
    {
        return EXIT_FAILURE;
    }

    std::cout << "conv2d backward test passed on "
              << kl::to_string(target.type())
              << '\n';

    return EXIT_SUCCESS;
}