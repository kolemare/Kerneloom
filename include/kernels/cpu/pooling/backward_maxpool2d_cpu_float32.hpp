#ifndef KL_BACKWARD_MAXPOOL2D_CPU_FLOAT32_HPP
#define KL_BACKWARD_MAXPOOL2D_CPU_FLOAT32_HPP

#include <core/tensor.hpp>

namespace kl
{

    void backward_maxpool2d_cpu_float32(
        const Tensor &indices,
        const Tensor &grad_output,
        Tensor &grad_input);

}

#endif // KL_BACKWARD_MAXPOOL2D_CPU_FLOAT32_HPP