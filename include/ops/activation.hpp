#ifndef KL_ACTIVATION_HPP
#define KL_ACTIVATION_HPP

#include <core/tensor.hpp>

namespace kl
{

    enum class ActivationType
    {
        ReLU,
        Sigmoid,
        Tanh,
        Softmax
    };

    const char *activation_type_name(
        ActivationType type);

    void activation(
        Tensor &tensor,
        ActivationType type);

    void activation_unchecked(
        Tensor &tensor,
        ActivationType type);

}

#endif // KL_ACTIVATION_HPP