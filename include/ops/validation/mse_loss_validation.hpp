#ifndef KL_MSE_LOSS_VALIDATION_HPP
#define KL_MSE_LOSS_VALIDATION_HPP

#include <ops/mse_loss.hpp>

#include <core/tensor.hpp>

#include <cstddef>

namespace kl
{

    void validate_mse_loss_inputs(
        const Tensor &prediction,
        const Tensor &target,
        const Tensor &result,
        Reduction reduction,
        std::size_t valid_sample_count);

}

#endif // KL_MSE_LOSS_VALIDATION_HPP