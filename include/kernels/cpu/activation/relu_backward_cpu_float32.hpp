#ifndef KL_RELU_BACKWARD_CPU_FLOAT32_HPP
#define KL_RELU_BACKWARD_CPU_FLOAT32_HPP

#include <core/tensor.hpp>

namespace kl
{

    void relu_backward_cpu_float32(
        const Tensor &activation_output,
        Tensor &grad);

}

#endif // KL_RELU_BACKWARD_CPU_FLOAT32_HPP