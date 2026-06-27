#ifndef KL_BINARY_CROSS_ENTROPY_HPP
#define KL_BINARY_CROSS_ENTROPY_HPP

#include <cnn/losses/reduction.hpp>

#include <core/tensor.hpp>

#include <cstddef>

namespace kl
{

    void binary_cross_entropy(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &result,
        Reduction reduction);

    void binary_cross_entropy(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &result,
        Reduction reduction,
        std::size_t valid_sample_count);

    void binary_cross_entropy_unchecked(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &result,
        Reduction reduction);

    void binary_cross_entropy_unchecked(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &result,
        Reduction reduction,
        std::size_t valid_sample_count);

}

#endif // KL_BINARY_CROSS_ENTROPY_HPP