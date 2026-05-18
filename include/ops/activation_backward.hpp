#ifndef KL_ACTIVATION_BACKWARD_HPP
#define KL_ACTIVATION_BACKWARD_HPP

#include <core/tensor.hpp>

#include <ops/activation.hpp>

namespace kl
{

    void activation_backward(
        const Tensor &activation_output,
        Tensor &grad,
        ActivationType type);

}

#endif // KL_ACTIVATION_BACKWARD_HPP