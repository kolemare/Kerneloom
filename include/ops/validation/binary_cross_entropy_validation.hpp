#ifndef KL_BINARY_CROSS_ENTROPY_VALIDATION_HPP
#define KL_BINARY_CROSS_ENTROPY_VALIDATION_HPP

#include <ops/binary_cross_entropy.hpp>

#include <core/tensor.hpp>

#include <cstddef>

namespace kl
{

    void validate_binary_cross_entropy_inputs(
        const Tensor &prediction,
        const Tensor &target,
        const Tensor &result,
        Reduction reduction,
        std::size_t valid_sample_count);

}

#endif // KL_BINARY_CROSS_ENTROPY_VALIDATION_HPP