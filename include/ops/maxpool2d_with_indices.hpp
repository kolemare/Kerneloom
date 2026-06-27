#ifndef KL_MAXPOOL2D_WITH_INDICES_HPP
#define KL_MAXPOOL2D_WITH_INDICES_HPP

#include <cnn/options/pooling2d_options.hpp>

#include <core/tensor.hpp>

namespace kl
{

    void maxpool2d_with_indices(
        const Tensor &input,
        Tensor &result,
        Tensor &indices,
        const Pooling2dOptions &options);

    void maxpool2d_with_indices_unchecked(
        const Tensor &input,
        Tensor &result,
        Tensor &indices,
        const Pooling2dOptions &options);

}

#endif // KL_MAXPOOL2D_WITH_INDICES_HPP