#ifndef KL_BACKWARD_ACTIVATION_HPP
#define KL_BACKWARD_ACTIVATION_HPP

#include <core/tensor.hpp>

#include <ops/activation.hpp>

namespace kl
{

    void backward_activation(
        const Tensor &activation_output,
        Tensor &grad,
        ActivationType type);

}

#endif // KL_BACKWARD_ACTIVATION_HPP