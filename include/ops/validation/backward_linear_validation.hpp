#ifndef KL_BACKWARD_LINEAR_VALIDATION_HPP
#define KL_BACKWARD_LINEAR_VALIDATION_HPP

#include <core/tensor.hpp>

namespace kl
{

    void validate_backward_linear_inputs(
        const Tensor &input,
        const Tensor &weights,
        const Tensor &grad_output,
        const Tensor &grad_input,
        const Tensor &grad_weights,
        const Tensor *grad_bias);

}

#endif // KL_BACKWARD_LINEAR_VALIDATION_HPP