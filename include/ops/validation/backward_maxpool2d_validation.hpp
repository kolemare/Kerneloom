#ifndef KL_BACKWARD_MAXPOOL2D_VALIDATION_HPP
#define KL_BACKWARD_MAXPOOL2D_VALIDATION_HPP

#include <core/tensor.hpp>

namespace kl
{

    void validate_backward_maxpool2d_inputs(
        const Tensor &indices,
        const Tensor &grad_output,
        const Tensor &grad_input);

}

#endif // KL_BACKWARD_MAXPOOL2D_VALIDATION_HPP