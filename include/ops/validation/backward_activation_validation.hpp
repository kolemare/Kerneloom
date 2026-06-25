#ifndef KL_BACKWARD_ACTIVATION_VALIDATION_HPP
#define KL_BACKWARD_ACTIVATION_VALIDATION_HPP

#include <ops/backward_activation.hpp>

#include <core/tensor.hpp>

namespace kl
{

    void validate_backward_activation_inputs(
        const Tensor &activation_output,
        const Tensor &grad,
        ActivationType type);

}

#endif // KL_BACKWARD_ACTIVATION_VALIDATION_HPP