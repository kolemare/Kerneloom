#ifndef KL_BACKWARD_ACTIVATION_HPP
#define KL_BACKWARD_ACTIVATION_HPP

#include <ops/activation.hpp>

#include <core/tensor.hpp>

namespace kl
{

    void backward_activation(
        const Tensor &activation_output,
        Tensor &grad,
        ActivationType type);

    void backward_activation_unchecked(
        const Tensor &activation_output,
        Tensor &grad,
        ActivationType type);

}

#endif // KL_BACKWARD_ACTIVATION_HPP