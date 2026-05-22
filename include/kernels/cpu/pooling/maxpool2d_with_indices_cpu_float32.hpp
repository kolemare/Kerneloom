#ifndef KL_MAXPOOL2D_WITH_INDICES_CPU_FLOAT32_HPP
#define KL_MAXPOOL2D_WITH_INDICES_CPU_FLOAT32_HPP

#include <cnn/options/pooling2d_options.hpp>

#include <core/tensor.hpp>

namespace kl
{

    void maxpool2d_with_indices_cpu_float32(
        const Tensor &input,
        Tensor &result,
        Tensor &indices,
        const Pooling2dOptions &options);

}

#endif // KL_MAXPOOL2D_WITH_INDICES_CPU_FLOAT32_HPP