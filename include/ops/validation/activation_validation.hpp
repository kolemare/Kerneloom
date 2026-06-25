#ifndef KL_ACTIVATION_VALIDATION_HPP
#define KL_ACTIVATION_VALIDATION_HPP

#include <ops/activation.hpp>

#include <core/tensor.hpp>

namespace kl
{

    void validate_activation_tensor(
        const Tensor &tensor,
        ActivationType type);

}

#endif // KL_ACTIVATION_VALIDATION_HPP