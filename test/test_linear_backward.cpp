#include <backend/backend.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/tensor.hpp>

#include <ops/backward_linear.hpp>

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

    kl::Tensor input_cpu(
        kl::Shape{2, 3},
        kl::DType::Float32,
        kl::Device::cpu(),
        kl::Layout::Unknown,
        kl::Storage::RowMajor);

    kl::Tensor weights_cpu(
        kl::Shape{4, 3},
        kl::DType::Float32,
        kl::Device::cpu(),
        kl::Layout::Unknown,
        kl::Storage::RowMajor);

    kl::Tensor grad_output_cpu(
        kl::Shape{2, 4},
        kl::DType::Float32,
        kl::Device::cpu(),
        kl::Layout::Unknown,
        kl::Storage::RowMajor);

    const float input_values[6] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f};

    const float weight_values[12] = {
        1.0f, 1.0f, 1.0f,
        2.0f, 2.0f, 2.0f,
        1.0f, 0.0f, 1.0f,
        0.0f, 1.0f, 0.0f};

    const float grad_output_values[8] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f};

    fill_tensor(input_cpu, input_values);
    fill_tensor(weights_cpu, weight_values);
    fill_tensor(grad_output_cpu, grad_output_values);

    kl::Tensor input = input_cpu.to(target);
    kl::Tensor weights = weights_cpu.to(target);
    kl::Tensor grad_output = grad_output_cpu.to(target);

    kl::Tensor grad_input(
        kl::Shape{2, 3},
        kl::DType::Float32,
        target,
        kl::Layout::Unknown,
        kl::Storage::RowMajor);

    kl::Tensor grad_weights(
        kl::Shape{4, 3},
        kl::DType::Float32,
        target,
        kl::Layout::Unknown,
        kl::Storage::RowMajor);

    kl::Tensor grad_bias(
        kl::Shape{4},
        kl::DType::Float32,
        target,
        kl::Layout::Unknown,
        kl::Storage::RowMajor);

    kl::backward_linear(
        input,
        weights,
        grad_output,
        grad_input,
        grad_weights,
        &grad_bias);

    kl::Tensor grad_input_cpu = grad_input.to(kl::Device::cpu());
    kl::Tensor grad_weights_cpu = grad_weights.to(kl::Device::cpu());
    kl::Tensor grad_bias_cpu = grad_bias.to(kl::Device::cpu());

    const float expected_grad_input[6] = {
        8.0f, 9.0f, 8.0f,
        24.0f, 25.0f, 24.0f};

    const float expected_grad_weights[12] = {
        21.0f, 27.0f, 33.0f,
        26.0f, 34.0f, 42.0f,
        31.0f, 41.0f, 51.0f,
        36.0f, 48.0f, 60.0f};

    const float expected_grad_bias[4] = {
        6.0f, 8.0f, 10.0f, 12.0f};

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

    std::cout << "linear backward test passed on "
              << kl::to_string(target.type())
              << '\n';

    return EXIT_SUCCESS;
}