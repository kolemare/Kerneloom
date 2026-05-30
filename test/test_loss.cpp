#include <backend/backend.hpp>

#include <cnn/losses/binary_cross_entropy_loss.hpp>
#include <cnn/losses/categorical_cross_entropy_loss.hpp>
#include <cnn/losses/loss.hpp>
#include <cnn/losses/mse_loss.hpp>
#include <cnn/losses/reduction.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/tensor.hpp>
#include <core/tensor_pool.hpp>

#include <cmath>
#include <cstdint>
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

    void fill_tensor(
        kl::Tensor &tensor,
        const std::int32_t *values)
    {
        std::int32_t *data =
            static_cast<std::int32_t *>(tensor.data());

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

    bool test_loss(
        kl::Device target,
        const char *name,
        std::unique_ptr<kl::Loss> loss,
        const float *prediction_values,
        const float *target_values,
        const float *expected_grad,
        std::size_t count,
        float expected_loss)
    {
        kl::Tensor prediction_cpu(
            kl::Shape{count},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::Unknown,
            kl::Storage::RowMajor);

        kl::Tensor target_cpu(
            kl::Shape{count},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::Unknown,
            kl::Storage::RowMajor);

        fill_tensor(
            prediction_cpu,
            prediction_values);

        fill_tensor(
            target_cpu,
            target_values);

        kl::Tensor prediction =
            prediction_cpu.to(target);

        kl::Tensor expected =
            target_cpu.to(target);

        kl::TensorPool pool;

        kl::Tensor &loss_value =
            loss->forward(
                prediction,
                expected,
                pool);

        kl::Tensor loss_value_cpu =
            loss_value.to(
                kl::Device::cpu());

        if (!check_tensor(
                loss_value_cpu,
                &expected_loss))
        {
            std::cout << name
                      << " forward failed\n";

            return false;
        }

        kl::Tensor &grad_prediction =
            loss->backward(pool);

        kl::Tensor grad_prediction_cpu =
            grad_prediction.to(
                kl::Device::cpu());

        if (!check_tensor(
                grad_prediction_cpu,
                expected_grad))
        {
            std::cout << name
                      << " backward failed\n";

            return false;
        }

        return true;
    }

    bool test_mse(
        kl::Device target)
    {
        const float prediction[4] = {
            1.0f, 2.0f, 3.0f, 4.0f};

        const float expected[4] = {
            0.0f, 2.0f, 4.0f, 2.0f};

        const float mean_grad[4] = {
            0.5f, 0.0f, -0.5f, 1.0f};

        const float sum_grad[4] = {
            2.0f, 0.0f, -2.0f, 4.0f};

        return test_loss(
                   target,
                   "MSE mean",
                   std::make_unique<kl::MSELoss>(
                       kl::Reduction::Mean),
                   prediction,
                   expected,
                   mean_grad,
                   4,
                   1.5f) &&
               test_loss(
                   target,
                   "MSE sum",
                   std::make_unique<kl::MSELoss>(
                       kl::Reduction::Sum),
                   prediction,
                   expected,
                   sum_grad,
                   4,
                   6.0f);
    }

    bool test_binary_cross_entropy(
        kl::Device target)
    {
        const float prediction[2] = {
            0.25f, 0.75f};

        const float expected[2] = {
            0.0f, 1.0f};

        const float mean_grad[2] = {
            0.6666667f, -0.6666667f};

        const float sum_grad[2] = {
            1.3333333f, -1.3333333f};

        return test_loss(
                   target,
                   "binary cross entropy mean",
                   std::make_unique<kl::BinaryCrossEntropyLoss>(
                       kl::Reduction::Mean),
                   prediction,
                   expected,
                   mean_grad,
                   2,
                   0.2876821f) &&
               test_loss(
                   target,
                   "binary cross entropy sum",
                   std::make_unique<kl::BinaryCrossEntropyLoss>(
                       kl::Reduction::Sum),
                   prediction,
                   expected,
                   sum_grad,
                   2,
                   0.5753641f);
    }

    bool test_categorical_cross_entropy(
        kl::Device target,
        kl::Reduction reduction,
        const float *expected_grad,
        float expected_loss)
    {
        kl::Tensor prediction_cpu(
            kl::Shape{2, 3},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::Unknown,
            kl::Storage::RowMajor);

        kl::Tensor target_cpu(
            kl::Shape{2},
            kl::DType::Int32,
            kl::Device::cpu(),
            kl::Layout::Unknown,
            kl::Storage::RowMajor);

        const float prediction_values[6] = {
            0.1f, 0.7f, 0.2f,
            0.6f, 0.3f, 0.1f};

        const std::int32_t target_values[2] = {
            1, 0};

        fill_tensor(
            prediction_cpu,
            prediction_values);

        fill_tensor(
            target_cpu,
            target_values);

        kl::Tensor prediction =
            prediction_cpu.to(target);

        kl::Tensor expected =
            target_cpu.to(target);

        kl::TensorPool pool;

        std::unique_ptr<kl::Loss> loss =
            std::make_unique<
                kl::CategoricalCrossEntropyLoss>(
                reduction);

        kl::Tensor &loss_value =
            loss->forward(
                prediction,
                expected,
                pool);

        kl::Tensor loss_value_cpu =
            loss_value.to(
                kl::Device::cpu());

        if (!check_tensor(
                loss_value_cpu,
                &expected_loss))
        {
            std::cout
                << "categorical cross entropy forward failed\n";

            return false;
        }

        kl::Tensor &grad_prediction =
            loss->backward(pool);

        kl::Tensor grad_prediction_cpu =
            grad_prediction.to(
                kl::Device::cpu());

        if (!check_tensor(
                grad_prediction_cpu,
                expected_grad))
        {
            std::cout
                << "categorical cross entropy backward failed\n";

            return false;
        }

        return true;
    }

    bool test_categorical_cross_entropy(
        kl::Device target)
    {
        const float mean_grad[6] = {
            0.0f, -0.7142857f, 0.0f,
            -0.8333333f, 0.0f, 0.0f};

        const float sum_grad[6] = {
            0.0f, -1.4285714f, 0.0f,
            -1.6666667f, 0.0f, 0.0f};

        return test_categorical_cross_entropy(
                   target,
                   kl::Reduction::Mean,
                   mean_grad,
                   0.4337503f) &&
               test_categorical_cross_entropy(
                   target,
                   kl::Reduction::Sum,
                   sum_grad,
                   0.8675006f);
    }

}

int main()
{
    const kl::Device target =
        kl::default_device();

    std::cout << "Target device: "
              << kl::to_string(target.type())
              << '\n';

    if (!test_mse(target) ||
        !test_binary_cross_entropy(target) ||
        !test_categorical_cross_entropy(target))
    {
        return EXIT_FAILURE;
    }

    std::cout << "loss tests passed on "
              << kl::to_string(target.type())
              << '\n';

    return EXIT_SUCCESS;
}