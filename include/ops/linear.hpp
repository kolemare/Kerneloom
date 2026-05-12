#ifndef KL_LINEAR_HPP
#define KL_LINEAR_HPP

#include <core/tensor.hpp>

namespace kl
{

    void linear(
        const Tensor &input,
        const Tensor &weights,
        const Tensor *bias,
        Tensor &result);

}

#endif // KL_LINEAR_HPP