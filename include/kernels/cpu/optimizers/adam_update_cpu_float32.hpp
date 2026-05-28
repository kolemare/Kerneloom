#ifndef KL_ADAM_UPDATE_CPU_FLOAT32_HPP
#define KL_ADAM_UPDATE_CPU_FLOAT32_HPP

#include <core/tensor.hpp>

namespace kl
{

    void adam_update_cpu_float32(
        Tensor &value,
        const Tensor &grad,
        Tensor &first_moment,
        Tensor &second_moment,
        float learning_rate,
        float beta1,
        float beta2,
        float epsilon,
        float beta1_power,
        float beta2_power);

}

#endif // KL_ADAM_UPDATE_CPU_FLOAT32_HPP