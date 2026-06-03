#include <cnn/losses/categorical_cross_entropy_loss.hpp>

#include <core/layout.hpp>
#include <core/shape.hpp>
#include <core/storage.hpp>

#include <ops/backward_categorical_cross_entropy.hpp>
#include <ops/categorical_cross_entropy.hpp>

#include <stdexcept>

namespace kl
{

    CategoricalCrossEntropyLoss::
        CategoricalCrossEntropyLoss(
            Reduction reduction)
        : reduction_(
              reduction)
    {
    }

    Tensor &
    CategoricalCrossEntropyLoss::forward(
        const Tensor &prediction,
        const Tensor &target,
        TensorPool &pool)
    {
        return forward(
            prediction,
            target,
            pool,
            prediction.shape()[0]);
    }

    Tensor &
    CategoricalCrossEntropyLoss::forward(
        const Tensor &prediction,
        const Tensor &target,
        TensorPool &pool,
        std::size_t valid_sample_count)
    {
        Tensor &result =
            pool.request(
                Shape{1},
                DType::Float32,
                prediction.device(),
                Layout::Unknown,
                Storage::RowMajor);

        categorical_cross_entropy(
            prediction,
            target,
            result,
            reduction_,
            valid_sample_count);

        last_prediction_ =
            &prediction;

        last_target_ =
            &target;

        last_valid_sample_count_ =
            valid_sample_count;

        return result;
    }

    Tensor &
    CategoricalCrossEntropyLoss::backward(
        TensorPool &pool)
    {
        if (last_prediction_ ==
                nullptr ||
            last_target_ ==
                nullptr)
        {
            throw std::runtime_error(
                "CategoricalCrossEntropyLoss::backward called before forward");
        }

        Tensor &grad_prediction =
            pool.request(
                last_prediction_
                    ->shape(),
                last_prediction_
                    ->dtype(),
                last_prediction_
                    ->device(),
                last_prediction_
                    ->layout(),
                last_prediction_
                    ->storage());

        backward_categorical_cross_entropy(
            *last_prediction_,
            *last_target_,
            grad_prediction,
            reduction_,
            last_valid_sample_count_);

        return grad_prediction;
    }

    Reduction
    CategoricalCrossEntropyLoss::reduction() const
    {
        return reduction_;
    }

}