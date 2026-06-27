#ifndef KL_BACKWARD_AVGPOOL2D_HPP
#define KL_BACKWARD_AVGPOOL2D_HPP

#include <cnn/options/pooling2d_options.hpp>
#include <core/tensor.hpp>

namespace kl
{

    void backward_avgpool2d(
        const Tensor &grad_output,
        Tensor &grad_input,
        const Pooling2dOptions &options);

    void backward_avgpool2d_unchecked(
        const Tensor &grad_output,
        Tensor &grad_input,
        const Pooling2dOptions &options);

}

#endif // KL_BACKWARD_AVGPOOL2D_HPP