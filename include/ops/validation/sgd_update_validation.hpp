#ifndef KL_SGD_UPDATE_VALIDATION_HPP
#define KL_SGD_UPDATE_VALIDATION_HPP

#include <core/tensor.hpp>

namespace kl
{

    void validate_sgd_update_inputs(
        const Tensor &value,
        const Tensor &grad);

}

#endif // KL_SGD_UPDATE_VALIDATION_HPP