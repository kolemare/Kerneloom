#ifndef KL_MSE_LOSS_CPU_FLOAT32_HPP
#define KL_MSE_LOSS_CPU_FLOAT32_HPP

#include <cnn/losses/reduction.hpp>

#include <core/tensor.hpp>

#include <cstddef>

namespace kl
{

    void mse_loss_cpu_float32(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &result,
        Reduction reduction,
        std::size_t valid_sample_count);

}

#endif // KL_MSE_LOSS_CPU_FLOAT32_HPP