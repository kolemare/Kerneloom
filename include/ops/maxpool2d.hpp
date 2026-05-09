#ifndef KL_MAXPOOL2D_HPP
#define KL_MAXPOOL2D_HPP

#include <cnn/options/pooling2d_options.hpp>
#include <core/tensor.hpp>

namespace kl
{

    void maxpool2d(
        const Tensor &input,
        Tensor &result,
        const Pooling2dOptions &options);

}

#endif // KL_MAXPOOL2D_HPP