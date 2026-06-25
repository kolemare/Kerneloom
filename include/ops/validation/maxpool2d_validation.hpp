#ifndef KL_MAXPOOL2D_VALIDATION_HPP
#define KL_MAXPOOL2D_VALIDATION_HPP

#include <cnn/options/pooling2d_options.hpp>
#include <core/tensor.hpp>

namespace kl
{

    void validate_maxpool2d_inputs(
        const Tensor &input,
        const Tensor &result,
        const Pooling2dOptions &options);

}

#endif // KL_MAXPOOL2D_VALIDATION_HPP