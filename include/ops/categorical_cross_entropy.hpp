#ifndef KL_CATEGORICAL_CROSS_ENTROPY_HPP
#define KL_CATEGORICAL_CROSS_ENTROPY_HPP

#include <cnn/losses/reduction.hpp>

#include <core/tensor.hpp>

namespace kl
{

    void categorical_cross_entropy(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &result,
        Reduction reduction);

}

#endif // KL_CATEGORICAL_CROSS_ENTROPY_HPP