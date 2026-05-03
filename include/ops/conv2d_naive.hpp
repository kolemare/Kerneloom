#ifndef KL_CONV2D_NAIVE_HPP
#define KL_CONV2D_NAIVE_HPP

#include <core/tensor.hpp>

namespace kl
{
    Tensor conv2d_naive(
        const Tensor &input,
        const Tensor &kernels);
}

#endif // KL_CONV2D_NAIVE_HPP