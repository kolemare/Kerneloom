#ifndef KL_PARAMETER_HPP
#define KL_PARAMETER_HPP

#include <core/tensor.hpp>

namespace kl
{

    struct Parameter
    {
        Tensor *value = nullptr;
        Tensor *grad = nullptr;
    };

}

#endif // KL_PARAMETER_HPP