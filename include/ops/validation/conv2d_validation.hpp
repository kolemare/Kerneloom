#ifndef KL_CONV2D_VALIDATION_HPP
#define KL_CONV2D_VALIDATION_HPP

#include <cnn/options/conv2d_options.hpp>
#include <core/tensor.hpp>

namespace kl
{

    void validate_conv2d_inputs(
        const Tensor &input,
        const Tensor &kernels,
        const Tensor *bias,
        const Tensor &result,
        const Conv2dOptions &options);

}

#endif // KL_CONV2D_VALIDATION_HPP