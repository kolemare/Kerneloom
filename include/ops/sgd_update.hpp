#ifndef KL_SGD_UPDATE_HPP
#define KL_SGD_UPDATE_HPP

#include <core/tensor.hpp>

namespace kl
{

    void sgd_update(
        Tensor &value,
        const Tensor &grad,
        float learning_rate);

    void sgd_update_unchecked(
        Tensor &value,
        const Tensor &grad,
        float learning_rate);

}

#endif // KL_SGD_UPDATE_HPP