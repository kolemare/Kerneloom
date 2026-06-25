#ifndef KL_BACKWARD_CONV2D_VALIDATION_HPP
#define KL_BACKWARD_CONV2D_VALIDATION_HPP

#include <cnn/options/conv2d_options.hpp>
#include <core/tensor.hpp>

namespace kl
{

    void validate_backward_conv2d_inputs(
        const Tensor &input,
        const Tensor &weights,
        const Tensor &grad_output,
        const Tensor &grad_input,
        const Tensor &grad_weights,
        const Tensor *grad_bias,
        const Conv2dOptions &options);

}

#endif // KL_BACKWARD_CONV2D_VALIDATION_HPP