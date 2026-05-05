#ifndef KL_MATMUL_NAIVE_HPP
#define KL_MATMUL_NAIVE_HPP

#include <core/tensor.hpp>

namespace kl
{

    void matmul_naive(
        const Tensor &a,
        const Tensor &b,
        Tensor &c);

}

#endif // KL_MATMUL_NAIVE_HPP