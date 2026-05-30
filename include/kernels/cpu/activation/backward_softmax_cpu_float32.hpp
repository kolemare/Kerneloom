#ifndef KL_BACKWARD_SOFTMAX_CPU_FLOAT32_HPP
#define KL_BACKWARD_SOFTMAX_CPU_FLOAT32_HPP

#include <core/tensor.hpp>

namespace kl
{

    void backward_softmax_cpu_float32(
        const Tensor &activation_output,
        Tensor &grad);

}

#endif // KL_BACKWARD_SOFTMAX_CPU_FLOAT32_HPP