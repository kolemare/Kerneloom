#ifndef KL_BINARY_CROSS_ENTROPY_HPP
#define KL_BINARY_CROSS_ENTROPY_HPP

#include <cnn/losses/reduction.hpp>

#include <core/tensor.hpp>

namespace kl
{

    void binary_cross_entropy(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &result,
        Reduction reduction);

}

#endif // KL_BINARY_CROSS_ENTROPY_HPP