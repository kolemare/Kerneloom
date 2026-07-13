#include <backend/backend.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>

#include <ops/activation.hpp>
#include <ops/backward_activation.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace
{

    void fill_tensor(
        kl::Tensor &tensor,
        const float *values)
    {
        float *data =
            static_cast<float *>(tensor.data());

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
        const float *data =
            static_cast<const float *>(tensor.data());

        for (std::size_t i = 0; i < tensor.numel(); ++i)
        {
            if (!close_enough(
                    data[i],
                    expected[i]))
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

    bool test_activation(
        kl::Device target,
        kl::ActivationType type,
        const kl::Shape &shape,
        const float *input_values,
        const float *incoming_grad_values,
        const float *expected_forward,
        const float *expected_backward)
    {
        kl::Tensor input_cpu(
            shape,
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::Unknown,
            kl::Storage::RowMajor);

        kl::Tensor grad_cpu(
            shape,
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::Unknown,
            kl::Storage::RowMajor);

        fill_tensor(
            input_cpu,
            input_values);

        fill_tensor(
            grad_cpu,
            incoming_grad_values);

        kl::Tensor tensor =
            input_cpu.to(target);

        kl::activation(
            tensor,
            type);

        kl::Tensor forward_cpu =
            tensor.to(kl::Device::cpu());

        if (!check_tensor(
                forward_cpu,
                expected_forward))
        {
            std::cout
                << kl::activation_type_name(type)
                << " forward failed\n";

            return false;
        }

        kl::Tensor grad =
            grad_cpu.to(target);

        kl::backward_activation(
            tensor,
            grad,
            type);

        kl::Tensor backward_cpu =
            grad.to(kl::Device::cpu());

        if (!check_tensor(
                backward_cpu,
                expected_backward))
        {
            std::cout
                << kl::activation_type_name(type)
                << " backward failed\n";

            return false;
        }

        return true;
    }

    bool test_relu(
        kl::Device target)
    {
        const float input[5] = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f};

        const float grad[5] = {
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

        const float expected_forward[5] = {
            0.0f, 0.0f, 0.0f, 1.0f, 2.0f};

        const float expected_backward[5] = {
            0.0f, 0.0f, 0.0f, 4.0f, 5.0f};

        return test_activation(
            target,
            kl::ActivationType::ReLU,
            kl::Shape{5},
            input,
            grad,
            expected_forward,
            expected_backward);
    }

    bool test_sigmoid(
        kl::Device target)
    {
        const float input[5] = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f};

        const float grad[5] = {
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

        const float expected_forward[5] = {
            0.1192029f,
            0.2689414f,
            0.5000000f,
            0.7310586f,
            0.8807971f};

        const float expected_backward[5] = {
            0.1049936f,
            0.3932239f,
            0.7500000f,
            0.7864477f,
            0.5249679f};

        return test_activation(
            target,
            kl::ActivationType::Sigmoid,
            kl::Shape{5},
            input,
            grad,
            expected_forward,
            expected_backward);
    }

    bool test_tanh(
        kl::Device target)
    {
        const float input[5] = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f};

        const float grad[5] = {
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

        const float expected_forward[5] = {
            -0.9640276f,
            -0.7615942f,
            0.0000000f,
            0.7615942f,
            0.9640276f};

        const float expected_backward[5] = {
            0.0706508f,
            0.8399487f,
            3.0000000f,
            1.6798974f,
            0.3532541f};

        return test_activation(
            target,
            kl::ActivationType::Tanh,
            kl::Shape{5},
            input,
            grad,
            expected_forward,
            expected_backward);
    }

    bool test_softmax(
        kl::Device target)
    {
        const float input[6] = {
            1.0f, 2.0f, 3.0f,
            0.0f, 0.0f, 0.0f};

        const float grad[6] = {
            1.0f, 2.0f, 3.0f,
            3.0f, 2.0f, 1.0f};

        const float expected_forward[6] = {
            0.0900306f,
            0.2447285f,
            0.6652410f,
            0.3333333f,
            0.3333333f,
            0.3333333f};

        const float expected_backward[6] = {
            -0.1418171f,
            -0.1407704f,
            0.2825875f,
            0.3333333f,
            0.0000000f,
            -0.3333333f};

        return test_activation(
            target,
            kl::ActivationType::Softmax,
            kl::Shape{2, 3},
            input,
            grad,
            expected_forward,
            expected_backward);
    }

}

int main()
{
    const kl::Device target =
        kl::default_device();

    std::cout
        << "Target device: "
        << kl::to_string(target.type())
        << '\n';

    if (!test_relu(target) ||
        !test_sigmoid(target) ||
        !test_tanh(target) ||
        !test_softmax(target))
    {
        return EXIT_FAILURE;
    }

    std::cout
        << "activation tests passed on "
        << kl::to_string(target.type())
        << '\n';

    return EXIT_SUCCESS;
}