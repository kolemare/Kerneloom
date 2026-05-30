#include <cnn/losses/mse_loss.hpp>

#include <core/layout.hpp>
#include <core/storage.hpp>

#include <ops/backward_mse_loss.hpp>
#include <ops/mse_loss.hpp>

#include <stdexcept>

namespace kl
{

    MSELoss::MSELoss(
        Reduction reduction)
        : reduction_(reduction)
    {
    }

    Tensor &MSELoss::forward(
        const Tensor &prediction,
        const Tensor &target,
        TensorPool &pool)
    {
        Tensor &result = pool.request(
            Shape{1},
            prediction.dtype(),
            prediction.device(),
            Layout::Unknown,
            Storage::RowMajor);

        mse_loss(
            prediction,
            target,
            result,
            reduction_);

        last_prediction_ = &prediction;
        last_target_ = &target;

        return result;
    }

    Tensor &MSELoss::backward(
        TensorPool &pool)
    {
        if (last_prediction_ == nullptr ||
            last_target_ == nullptr)
        {
            throw std::runtime_error("MSELoss::backward called before forward");
        }

        Tensor &grad_prediction = pool.request(
            last_prediction_->shape(),
            last_prediction_->dtype(),
            last_prediction_->device(),
            last_prediction_->layout(),
            last_prediction_->storage());

        backward_mse_loss(
            *last_prediction_,
            *last_target_,
            grad_prediction,
            reduction_);

        return grad_prediction;
    }

    Reduction MSELoss::reduction() const
    {
        return reduction_;
    }

}