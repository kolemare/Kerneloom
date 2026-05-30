#include <cnn/losses/loss.hpp>
#include <cnn/losses/mse_loss.hpp>
#include <cnn/losses/reduction.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/tensor.hpp>
#include <core/tensor_pool.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>

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

    bool check_scalar(
        const kl::Tensor &tensor,
        float expected)
    {
        if (tensor.numel() != 1)
        {
            std::cout << "expected scalar tensor\n";
            return false;
        }

        const float *data =
            static_cast<const float *>(tensor.data());

        if (!close_enough(data[0], expected))
        {
            std::cout << "scalar mismatch"
                      << " | actual=" << data[0]
                      << " | expected=" << expected
                      << '\n';

            return false;
        }

        return true;
    }

    bool check_tensor(
        const kl::Tensor &tensor,
        const float *expected)
    {
        const float *data =
            static_cast<const float *>(tensor.data());

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

    bool test_mse_mean()
    {
        kl::Tensor prediction(
            kl::Shape{4},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::Unknown,
            kl::Storage::RowMajor);

        kl::Tensor target(
            kl::Shape{4},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::Unknown,
            kl::Storage::RowMajor);

        const float prediction_values[4] = {
            1.0f, 2.0f, 3.0f, 4.0f};

        const float target_values[4] = {
            0.0f, 2.0f, 4.0f, 2.0f};

        fill_tensor(
            prediction,
            prediction_values);

        fill_tensor(
            target,
            target_values);

        kl::TensorPool pool;

        std::unique_ptr<kl::Loss> loss =
            std::make_unique<kl::MSELoss>(
                kl::Reduction::Mean);

        kl::Tensor &loss_value =
            loss->forward(
                prediction,
                target,
                pool);

        if (!check_scalar(loss_value, 1.5f))
        {
            std::cout << "MSE mean forward failed\n";
            return false;
        }

        kl::Tensor &grad_prediction =
            loss->backward(pool);

        const float expected_grad[4] = {
            0.5f, 0.0f, -0.5f, 1.0f};

        if (!check_tensor(
                grad_prediction,
                expected_grad))
        {
            std::cout << "MSE mean backward failed\n";
            return false;
        }

        return true;
    }

    bool test_mse_sum()
    {
        kl::Tensor prediction(
            kl::Shape{4},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::Unknown,
            kl::Storage::RowMajor);

        kl::Tensor target(
            kl::Shape{4},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::Unknown,
            kl::Storage::RowMajor);

        const float prediction_values[4] = {
            1.0f, 2.0f, 3.0f, 4.0f};

        const float target_values[4] = {
            0.0f, 2.0f, 4.0f, 2.0f};

        fill_tensor(
            prediction,
            prediction_values);

        fill_tensor(
            target,
            target_values);

        kl::TensorPool pool;

        std::unique_ptr<kl::Loss> loss =
            std::make_unique<kl::MSELoss>(
                kl::Reduction::Sum);

        kl::Tensor &loss_value =
            loss->forward(
                prediction,
                target,
                pool);

        if (!check_scalar(loss_value, 6.0f))
        {
            std::cout << "MSE sum forward failed\n";
            return false;
        }

        kl::Tensor &grad_prediction =
            loss->backward(pool);

        const float expected_grad[4] = {
            2.0f, 0.0f, -2.0f, 4.0f};

        if (!check_tensor(
                grad_prediction,
                expected_grad))
        {
            std::cout << "MSE sum backward failed\n";
            return false;
        }

        return true;
    }

}

int main()
{
    bool passed = true;

    if (!test_mse_mean())
    {
        passed = false;
    }

    if (!test_mse_sum())
    {
        passed = false;
    }

    if (!passed)
    {
        return EXIT_FAILURE;
    }

    std::cout << "loss tests passed on CPU\n";

    return EXIT_SUCCESS;
}