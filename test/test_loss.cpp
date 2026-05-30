#include <cnn/losses/binary_cross_entropy_loss.hpp>
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

    bool test_loss(
        const char *name,
        std::unique_ptr<kl::Loss> loss,
        const float *prediction_values,
        const float *target_values,
        const float *expected_grad,
        std::size_t count,
        float expected_loss)
    {
        kl::Tensor prediction(
            kl::Shape{count},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::Unknown,
            kl::Storage::RowMajor);

        kl::Tensor target(
            kl::Shape{count},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::Unknown,
            kl::Storage::RowMajor);

        fill_tensor(
            prediction,
            prediction_values);

        fill_tensor(
            target,
            target_values);

        kl::TensorPool pool;

        kl::Tensor &loss_value =
            loss->forward(
                prediction,
                target,
                pool);

        if (!check_tensor(
                loss_value,
                &expected_loss))
        {
            std::cout << name << " forward failed\n";
            return false;
        }

        kl::Tensor &grad_prediction =
            loss->backward(pool);

        if (!check_tensor(
                grad_prediction,
                expected_grad))
        {
            std::cout << name << " backward failed\n";
            return false;
        }

        return true;
    }

    bool test_mse()
    {
        const float prediction[4] = {
            1.0f, 2.0f, 3.0f, 4.0f};

        const float target[4] = {
            0.0f, 2.0f, 4.0f, 2.0f};

        const float expected_mean_grad[4] = {
            0.5f, 0.0f, -0.5f, 1.0f};

        const float expected_sum_grad[4] = {
            2.0f, 0.0f, -2.0f, 4.0f};

        return test_loss(
                   "MSE mean",
                   std::make_unique<kl::MSELoss>(
                       kl::Reduction::Mean),
                   prediction,
                   target,
                   expected_mean_grad,
                   4,
                   1.5f) &&
               test_loss(
                   "MSE sum",
                   std::make_unique<kl::MSELoss>(
                       kl::Reduction::Sum),
                   prediction,
                   target,
                   expected_sum_grad,
                   4,
                   6.0f);
    }

    bool test_binary_cross_entropy()
    {
        const float prediction[2] = {
            0.25f, 0.75f};

        const float target[2] = {
            0.0f, 1.0f};

        const float expected_mean_grad[2] = {
            0.6666667f, -0.6666667f};

        const float expected_sum_grad[2] = {
            1.3333333f, -1.3333333f};

        return test_loss(
                   "binary cross entropy mean",
                   std::make_unique<kl::BinaryCrossEntropyLoss>(
                       kl::Reduction::Mean),
                   prediction,
                   target,
                   expected_mean_grad,
                   2,
                   0.2876821f) &&
               test_loss(
                   "binary cross entropy sum",
                   std::make_unique<kl::BinaryCrossEntropyLoss>(
                       kl::Reduction::Sum),
                   prediction,
                   target,
                   expected_sum_grad,
                   2,
                   0.5753641f);
    }

}

int main()
{
    if (!test_mse() ||
        !test_binary_cross_entropy())
    {
        return EXIT_FAILURE;
    }

    std::cout << "loss tests passed on CPU\n";

    return EXIT_SUCCESS;
}