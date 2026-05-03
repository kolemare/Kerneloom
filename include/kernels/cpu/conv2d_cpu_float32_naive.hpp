#ifndef KL_CONV2D_CPU_FLOAT32_NAIVE_HPP
#define KL_CONV2D_CPU_FLOAT32_NAIVE_HPP

#include <core/tensor.hpp>

namespace kl
{

    void conv2d_cpu_float32_naive(
        const Tensor &input,
        const Tensor &kernels,
        Tensor &result);

}

#endif // KL_CONV2D_CPU_FLOAT32_NAIVE_HPP