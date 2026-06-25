#ifndef KL_ADAM_UPDATE_VALIDATION_HPP
#define KL_ADAM_UPDATE_VALIDATION_HPP

#include <core/tensor.hpp>

namespace kl
{

    void validate_adam_update_inputs(
        const Tensor &value,
        const Tensor &grad,
        const Tensor &first_moment,
        const Tensor &second_moment,
        float learning_rate,
        float beta1,
        float beta2,
        float epsilon,
        float beta1_power,
        float beta2_power);

}

#endif // KL_ADAM_UPDATE_VALIDATION_HPP