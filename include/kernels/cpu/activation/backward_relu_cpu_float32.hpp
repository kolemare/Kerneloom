#ifndef KL_BACKWARD_RELU_CPU_FLOAT32_HPP
#define KL_BACKWARD_RELU_CPU_FLOAT32_HPP

#include <core/tensor.hpp>

namespace kl
{

    void backward_relu_cpu_float32(
        const Tensor &activation_output,
        Tensor &grad);

}

#endif // KL_BACKWARD_RELU_CPU_FLOAT32_HPP