#include <backend/backend.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/tensor.hpp>

#include <ops/activation.hpp>
#include <ops/backward_activation.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace
{

    void fill_tensor(kl::Tensor &tensor, const float *values)
    {
        float *data = static_cast<float *>(tensor.data());

        for (std::size_t i = 0; i < tensor.numel(); ++i)
        {
            data[i] = values[i];
        }
    }

    bool close_enough(float actual, float expected)
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
                return false;
            }
        }

        return true;
    }

    bool run_activation_backward_test(
        kl::Device target,
        kl::ActivationType type,
        const float *activation_output_values,
        const float *grad_values,
        const float *expected_values,
        std::size_t count)
    {
        kl::Tensor activation_output_cpu(
            kl::Shape{count},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::Unknown,
            kl::Storage::RowMajor);

        kl::Tensor grad_cpu(
            kl::Shape{count},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::Unknown,
            kl::Storage::RowMajor);

        fill_tensor(activation_output_cpu, activation_output_values);
        fill_tensor(grad_cpu, grad_values);

        kl::Tensor activation_output = activation_output_cpu.to(target);
        kl::Tensor grad = grad_cpu.to(target);

        kl::backward_activation(
            activation_output,
            grad,
            type);

        kl::Tensor result_cpu = grad.to(kl::Device::cpu());

        return check_tensor(result_cpu, expected_values);
    }

    bool test_relu(kl::Device target)
    {
        const float output_values[5] = {
            0.0f, 1.0f, 0.0f, 3.0f, 5.0f};

        const float grad_values[5] = {
            1.0f, 1.0f, 2.0f, 2.0f, 3.0f};

        const float expected[5] = {
            0.0f, 1.0f, 0.0f, 2.0f, 3.0f};

        return run_activation_backward_test(
            target,
            kl::ActivationType::ReLU,
            output_values,
            grad_values,
            expected,
            5);
    }

    bool test_sigmoid(kl::Device target)
    {
        const float output_values[4] = {
            0.25f, 0.5f, 0.75f, 1.0f};

        const float grad_values[4] = {
            4.0f, 4.0f, 4.0f, 4.0f};

        const float expected[4] = {
            0.75f, 1.0f, 0.75f, 0.0f};

        return run_activation_backward_test(
            target,
            kl::ActivationType::Sigmoid,
            output_values,
            grad_values,
            expected,
            4);
    }

    bool test_tanh(kl::Device target)
    {
        const float output_values[4] = {
            0.0f, 0.5f, -0.5f, 1.0f};

        const float grad_values[4] = {
            2.0f, 2.0f, 4.0f, 8.0f};

        const float expected[4] = {
            2.0f, 1.5f, 3.0f, 0.0f};

        return run_activation_backward_test(
            target,
            kl::ActivationType::Tanh,
            output_values,
            grad_values,
            expected,
            4);
    }

}

int main()
{
    const kl::Device target = kl::default_device();

    bool passed = true;

    if (!test_relu(target))
    {
        std::cout << "relu backward failed\n";
        passed = false;
    }

    if (!test_sigmoid(target))
    {
        std::cout << "sigmoid backward failed\n";
        passed = false;
    }

    if (!test_tanh(target))
    {
        std::cout << "tanh backward failed\n";
        passed = false;
    }

    if (!passed)
    {
        return EXIT_FAILURE;
    }

    std::cout << "activation backward test passed on "
              << kl::to_string(target.type())
              << '\n';

    return EXIT_SUCCESS;
}