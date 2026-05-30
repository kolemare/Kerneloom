#ifndef KL_BACKWARD_CATEGORICAL_CROSS_ENTROPY_HPP
#define KL_BACKWARD_CATEGORICAL_CROSS_ENTROPY_HPP

#include <cnn/losses/reduction.hpp>

#include <core/tensor.hpp>

namespace kl
{

    void backward_categorical_cross_entropy(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &grad_prediction,
        Reduction reduction);

}

#endif // KL_BACKWARD_CATEGORICAL_CROSS_ENTROPY_HPP