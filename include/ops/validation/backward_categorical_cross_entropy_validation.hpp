#ifndef KL_BACKWARD_CATEGORICAL_CROSS_ENTROPY_VALIDATION_HPP
#define KL_BACKWARD_CATEGORICAL_CROSS_ENTROPY_VALIDATION_HPP

#include <ops/backward_categorical_cross_entropy.hpp>

#include <core/tensor.hpp>

#include <cstddef>

namespace kl
{

    void validate_backward_categorical_cross_entropy_inputs(
        const Tensor &prediction,
        const Tensor &target,
        const Tensor &grad_prediction,
        Reduction reduction,
        std::size_t valid_sample_count);

}

#endif // KL_BACKWARD_CATEGORICAL_CROSS_ENTROPY_VALIDATION_HPP