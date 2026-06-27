#ifndef KL_BACKWARD_BINARY_CROSS_ENTROPY_HPP
#define KL_BACKWARD_BINARY_CROSS_ENTROPY_HPP

#include <cnn/losses/reduction.hpp>

#include <core/tensor.hpp>

#include <cstddef>

namespace kl
{

    void backward_binary_cross_entropy(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &grad_prediction,
        Reduction reduction);

    void backward_binary_cross_entropy(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &grad_prediction,
        Reduction reduction,
        std::size_t valid_sample_count);

    void backward_binary_cross_entropy_unchecked(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &grad_prediction,
        Reduction reduction);

    void backward_binary_cross_entropy_unchecked(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &grad_prediction,
        Reduction reduction,
        std::size_t valid_sample_count);

}

#endif // KL_BACKWARD_BINARY_CROSS_ENTROPY_HPP