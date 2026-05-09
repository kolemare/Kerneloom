#ifndef KL_AVGPOOL2D_CPU_FLOAT32_HPP
#define KL_AVGPOOL2D_CPU_FLOAT32_HPP

#include <cnn/options/pooling2d_options.hpp>
#include <core/tensor.hpp>

namespace kl
{

    void avgpool2d_cpu_float32(
        const Tensor &input,
        Tensor &result,
        const Pooling2dOptions &options);

}

#endif // KL_AVGPOOL2D_CPU_FLOAT32_HPP