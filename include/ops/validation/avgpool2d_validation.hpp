#ifndef KL_AVGPOOL2D_VALIDATION_HPP
#define KL_AVGPOOL2D_VALIDATION_HPP

#include <cnn/options/pooling2d_options.hpp>
#include <core/tensor.hpp>

namespace kl
{

    void validate_avgpool2d_inputs(
        const Tensor &input,
        const Tensor &result,
        const Pooling2dOptions &options);

}

#endif // KL_AVGPOOL2D_VALIDATION_HPP