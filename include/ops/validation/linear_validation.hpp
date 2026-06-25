#ifndef KL_LINEAR_VALIDATION_HPP
#define KL_LINEAR_VALIDATION_HPP

#include <core/tensor.hpp>

namespace kl
{

    void validate_linear_inputs(
        const Tensor &input,
        const Tensor &weights,
        const Tensor *bias,
        const Tensor &result);

}

#endif // KL_LINEAR_VALIDATION_HPP