#ifndef KL_BACKWARD_AVGPOOL2D_VALIDATION_HPP
#define KL_BACKWARD_AVGPOOL2D_VALIDATION_HPP

#include <cnn/options/pooling2d_options.hpp>
#include <core/tensor.hpp>

namespace kl
{

    void validate_backward_avgpool2d_inputs(
        const Tensor &grad_output,
        const Tensor &grad_input,
        const Pooling2dOptions &options);

}

#endif // KL_BACKWARD_AVGPOOL2D_VALIDATION_HPP