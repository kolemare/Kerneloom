#ifndef KL_CATEGORICAL_CROSS_ENTROPY_VALIDATION_HPP
#define KL_CATEGORICAL_CROSS_ENTROPY_VALIDATION_HPP

#include <ops/categorical_cross_entropy.hpp>

#include <core/tensor.hpp>

#include <cstddef>

namespace kl
{

    void validate_categorical_cross_entropy_inputs(
        const Tensor &prediction,
        const Tensor &target,
        const Tensor &result,
        Reduction reduction,
        std::size_t valid_sample_count);

}

#endif // KL_CATEGORICAL_CROSS_ENTROPY_VALIDATION_HPP