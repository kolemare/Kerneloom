#ifndef KL_BACKWARD_CONV2D_HPP
#define KL_BACKWARD_CONV2D_HPP

#include <cnn/options/conv2d_options.hpp>

#include <core/tensor.hpp>

namespace kl
{

    void backward_conv2d(
        const Tensor &input,
        const Tensor &weights,
        const Tensor &grad_output,
        Tensor &grad_input,
        Tensor &grad_weights,
        Tensor *grad_bias,
        const Conv2dOptions &options);

}

#endif // KL_BACKWARD_CONV2D_HPP