#ifndef KL_BACKWARD_MSE_LOSS_CPU_FLOAT32_HPP
#define KL_BACKWARD_MSE_LOSS_CPU_FLOAT32_HPP

#include <cnn/losses/reduction.hpp>

#include <core/tensor.hpp>

namespace kl
{

    void backward_mse_loss_cpu_float32(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &grad_prediction,
        Reduction reduction);

}

#endif // KL_BACKWARD_MSE_LOSS_CPU_FLOAT32_HPP