#ifndef KL_CONV2D_CPU_FLOAT32_GENERAL_HPP
#define KL_CONV2D_CPU_FLOAT32_GENERAL_HPP

#include <cnn/options/conv2d_options.hpp>
#include <core/tensor.hpp>

namespace kl
{

    void conv2d_cpu_float32_general(
        const Tensor &input,
        const Tensor &kernels,
        const Tensor *bias,
        Tensor &result,
        const Conv2dOptions &options);

}

#endif // KL_CONV2D_CPU_FLOAT32_GENERAL_HPP