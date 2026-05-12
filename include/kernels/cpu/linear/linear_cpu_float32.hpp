#ifndef KL_LINEAR_CPU_FLOAT32_HPP
#define KL_LINEAR_CPU_FLOAT32_HPP

#include <core/tensor.hpp>

namespace kl
{

    void linear_cpu_float32(
        const Tensor &input,
        const Tensor &weights,
        const Tensor *bias,
        Tensor &result);

}

#endif // KL_LINEAR_CPU_FLOAT32_HPP