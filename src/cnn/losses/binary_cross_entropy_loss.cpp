#include <cnn/losses/binary_cross_entropy_loss.hpp>

#include <core/layout.hpp>
#include <core/storage.hpp>

#include <ops/backward_binary_cross_entropy.hpp>
#include <ops/binary_cross_entropy.hpp>

#include <stdexcept>

namespace kl
{

    BinaryCrossEntropyLoss::
        BinaryCrossEntropyLoss(
            Reduction reduction)
        : reduction_(
              reduction)
    {
    }

    Tensor &
    BinaryCrossEntropyLoss::forward(
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
    BinaryCrossEntropyLoss::forward(
        const Tensor &prediction,
        const Tensor &target,
        TensorPool &pool,
        std::size_t valid_sample_count)
    {
        Tensor &result =
            pool.request(
                Shape{1},
                prediction.dtype(),
                prediction.device(),
                Layout::Unknown,
                Storage::RowMajor);

        binary_cross_entropy_unchecked(
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
    BinaryCrossEntropyLoss::backward(
        TensorPool &pool)
    {
        if (last_prediction_ ==
                nullptr ||
            last_target_ ==
                nullptr)
        {
            throw std::runtime_error(
                "BinaryCrossEntropyLoss::backward called before forward");
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

        backward_binary_cross_entropy_unchecked(
            *last_prediction_,
            *last_target_,
            grad_prediction,
            reduction_,
            last_valid_sample_count_);

        return grad_prediction;
    }

    Reduction
    BinaryCrossEntropyLoss::reduction() const
    {
        return reduction_;
    }

}