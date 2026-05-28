#include <backend/backend.hpp>

#include <cnn/optimizers/parameter.hpp>
#include <cnn/optimizers/sgd_optimizer.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/tensor.hpp>

#include <ops/adam_update.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

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

    void fill_tensor(
        kl::Tensor &tensor,
        float value)
    {
        float *data = static_cast<float *>(tensor.data());

        for (std::size_t i = 0; i < tensor.numel(); ++i)
        {
            data[i] = value;
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

    bool test_sgd(kl::Device target)
    {
        kl::Tensor value_cpu(
            kl::Shape{4},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::Unknown,
            kl::Storage::RowMajor);

        kl::Tensor grad_cpu(
            kl::Shape{4},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::Unknown,
            kl::Storage::RowMajor);

        const float value_values[4] = {
            1.0f, 2.0f, 3.0f, 4.0f};

        const float grad_values[4] = {
            0.1f, 0.2f, 0.3f, 0.4f};

        fill_tensor(value_cpu, value_values);
        fill_tensor(grad_cpu, grad_values);

        kl::Tensor value = value_cpu.to(target);
        kl::Tensor grad = grad_cpu.to(target);

        std::vector<kl::Parameter> parameters;
        parameters.push_back(kl::Parameter{
            &value,
            &grad});

        kl::SGDOptimizer optimizer(0.5f);

        optimizer.step(parameters);

        kl::Tensor result_cpu = value.to(kl::Device::cpu());

        const float expected_after_first_step[4] = {
            0.95f, 1.90f, 2.85f, 3.80f};

        if (!check_tensor(result_cpu, expected_after_first_step))
        {
            std::cout << "SGD first step failed\n";
            return false;
        }

        optimizer.step(parameters);

        result_cpu = value.to(kl::Device::cpu());

        const float expected_after_second_step[4] = {
            0.90f, 1.80f, 2.70f, 3.60f};

        if (!check_tensor(result_cpu, expected_after_second_step))
        {
            std::cout << "SGD second step failed\n";
            return false;
        }

        return true;
    }

    bool test_adam(kl::Device target)
    {
        kl::Tensor value_cpu(
            kl::Shape{4},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::Unknown,
            kl::Storage::RowMajor);

        kl::Tensor grad_cpu(
            kl::Shape{4},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::Unknown,
            kl::Storage::RowMajor);

        kl::Tensor first_moment_cpu(
            kl::Shape{4},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::Unknown,
            kl::Storage::RowMajor);

        kl::Tensor second_moment_cpu(
            kl::Shape{4},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::Unknown,
            kl::Storage::RowMajor);

        const float value_values[4] = {
            1.0f, 2.0f, 3.0f, 4.0f};

        const float grad_values[4] = {
            0.1f, 0.2f, 0.3f, 0.4f};

        fill_tensor(value_cpu, value_values);
        fill_tensor(grad_cpu, grad_values);
        fill_tensor(first_moment_cpu, 0.0f);
        fill_tensor(second_moment_cpu, 0.0f);

        kl::Tensor value = value_cpu.to(target);
        kl::Tensor grad = grad_cpu.to(target);
        kl::Tensor first_moment = first_moment_cpu.to(target);
        kl::Tensor second_moment = second_moment_cpu.to(target);

        const float learning_rate = 0.001f;
        const float beta1 = 0.9f;
        const float beta2 = 0.999f;
        const float epsilon = 1.0e-8f;

        kl::adam_update(
            value,
            grad,
            first_moment,
            second_moment,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            beta1,
            beta2);

        kl::Tensor result_cpu = value.to(kl::Device::cpu());

        const float expected_after_first_step[4] = {
            0.999f, 1.999f, 2.999f, 3.999f};

        if (!check_tensor(result_cpu, expected_after_first_step))
        {
            std::cout << "Adam first step failed\n";
            return false;
        }

        kl::adam_update(
            value,
            grad,
            first_moment,
            second_moment,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            beta1 * beta1,
            beta2 * beta2);

        result_cpu = value.to(kl::Device::cpu());

        const float expected_after_second_step[4] = {
            0.998f, 1.998f, 2.998f, 3.998f};

        if (!check_tensor(result_cpu, expected_after_second_step))
        {
            std::cout << "Adam second step failed\n";
            return false;
        }

        return true;
    }

}

int main()
{
    const kl::Device target = kl::default_device();

    bool passed = true;

    if (!test_sgd(target))
    {
        passed = false;
    }

    if (!test_adam(target))
    {
        passed = false;
    }

    if (!passed)
    {
        return EXIT_FAILURE;
    }

    std::cout << "optimizer tests passed on "
              << kl::to_string(target.type())
              << '\n';

    return EXIT_SUCCESS;
}