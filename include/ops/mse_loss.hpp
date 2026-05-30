#ifndef KL_MSE_LOSS_OP_HPP
#define KL_MSE_LOSS_OP_HPP

#include <cnn/losses/reduction.hpp>

#include <core/tensor.hpp>

namespace kl
{

    void mse_loss(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &result,
        Reduction reduction);

}

#endif // KL_MSE_LOSS_OP_HPP