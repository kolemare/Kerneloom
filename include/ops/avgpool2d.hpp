#ifndef KL_AVGPOOL2D_HPP
#define KL_AVGPOOL2D_HPP

#include <cnn/options/pooling2d_options.hpp>
#include <core/tensor.hpp>

namespace kl
{

    void avgpool2d(
        const Tensor &input,
        Tensor &result,
        const Pooling2dOptions &options);

}

#endif // KL_AVGPOOL2D_HPP