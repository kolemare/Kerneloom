#ifndef KL_BACKWARD_MSE_LOSS_HPP
#define KL_BACKWARD_MSE_LOSS_HPP

#include <cnn/losses/reduction.hpp>

#include <core/tensor.hpp>

#include <cstddef>

namespace kl
{

    void backward_mse_loss(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &grad_prediction,
        Reduction reduction);

    void backward_mse_loss(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &grad_prediction,
        Reduction reduction,
        std::size_t valid_sample_count);

    void backward_mse_loss_unchecked(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &grad_prediction,
        Reduction reduction);

    void backward_mse_loss_unchecked(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &grad_prediction,
        Reduction reduction,
        std::size_t valid_sample_count);

}

#endif // KL_BACKWARD_MSE_LOSS_HPP