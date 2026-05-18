#include <backend/backend.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/tensor.hpp>

#include <ops/activation.hpp>
#include <ops/activation_backward.hpp>

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

    bool test_relu()
    {
        kl::Tensor activation_output(
            kl::Shape{5},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::Unknown,
            kl::Storage::RowMajor);

        kl::Tensor grad(
            kl::Shape{5},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::Unknown,
            kl::Storage::RowMajor);

        const float output_values[5] = {
            0.0f, 1.0f, 0.0f, 3.0f, 5.0f};

        const float grad_values[5] = {
            1.0f, 1.0f, 2.0f, 2.0f, 3.0f};

        const float expected[5] = {
            0.0f, 1.0f, 0.0f, 2.0f, 3.0f};

        fill_tensor(activation_output, output_values);
        fill_tensor(grad, grad_values);

        kl::activation_backward(
            activation_output,
            grad,
            kl::ActivationType::ReLU);

        return check_tensor(grad, expected);
    }

    bool test_sigmoid()
    {
        kl::Tensor activation_output(
            kl::Shape{4},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::Unknown,
            kl::Storage::RowMajor);

        kl::Tensor grad(
            kl::Shape{4},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::Unknown,
            kl::Storage::RowMajor);

        const float output_values[4] = {
            0.25f, 0.5f, 0.75f, 1.0f};

        const float grad_values[4] = {
            4.0f, 4.0f, 4.0f, 4.0f};

        const float expected[4] = {
            0.75f, 1.0f, 0.75f, 0.0f};

        fill_tensor(activation_output, output_values);
        fill_tensor(grad, grad_values);

        kl::activation_backward(
            activation_output,
            grad,
            kl::ActivationType::Sigmoid);

        return check_tensor(grad, expected);
    }

    bool test_tanh()
    {
        kl::Tensor activation_output(
            kl::Shape{4},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::Unknown,
            kl::Storage::RowMajor);

        kl::Tensor grad(
            kl::Shape{4},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::Unknown,
            kl::Storage::RowMajor);

        const float output_values[4] = {
            0.0f, 0.5f, -0.5f, 1.0f};

        const float grad_values[4] = {
            2.0f, 2.0f, 4.0f, 8.0f};

        const float expected[4] = {
            2.0f, 1.5f, 3.0f, 0.0f};

        fill_tensor(activation_output, output_values);
        fill_tensor(grad, grad_values);

        kl::activation_backward(
            activation_output,
            grad,
            kl::ActivationType::Tanh);

        return check_tensor(grad, expected);
    }

}

int main()
{
    bool passed = true;

    if (!test_relu())
    {
        std::cout << "relu backward failed\n";
        passed = false;
    }

    if (!test_sigmoid())
    {
        std::cout << "sigmoid backward failed\n";
        passed = false;
    }

    if (!test_tanh())
    {
        std::cout << "tanh backward failed\n";
        passed = false;
    }

    if (!passed)
    {
        return EXIT_FAILURE;
    }

    std::cout << "activation backward cpu test passed\n";

    return EXIT_SUCCESS;
}