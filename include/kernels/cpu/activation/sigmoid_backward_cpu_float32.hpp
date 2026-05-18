#ifndef KL_SIGMOID_BACKWARD_CPU_FLOAT32_HPP
#define KL_SIGMOID_BACKWARD_CPU_FLOAT32_HPP

#include <core/tensor.hpp>

namespace kl
{

    void sigmoid_backward_cpu_float32(
        const Tensor &activation_output,
        Tensor &grad);

}

#endif // KL_SIGMOID_BACKWARD_CPU_FLOAT32_HPP