#ifndef KL_ADAM_UPDATE_HPP
#define KL_ADAM_UPDATE_HPP

#include <core/tensor.hpp>

namespace kl
{

    void adam_update(
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

    void adam_update_unchecked(
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

#endif // KL_ADAM_UPDATE_HPP