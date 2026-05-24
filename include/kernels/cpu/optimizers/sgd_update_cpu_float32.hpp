#ifndef KL_SGD_UPDATE_CPU_FLOAT32_HPP
#define KL_SGD_UPDATE_CPU_FLOAT32_HPP

#include <core/tensor.hpp>

namespace kl
{

    void sgd_update_cpu_float32(
        Tensor &value,
        const Tensor &grad,
        float learning_rate);

}

#endif // KL_SGD_UPDATE_CPU_FLOAT32_HPP