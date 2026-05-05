#ifndef KL_MATMUL_CPU_FLOAT32_NAIVE_HPP
#define KL_MATMUL_CPU_FLOAT32_NAIVE_HPP

#include <core/tensor.hpp>

namespace kl
{

    void matmul_cpu_float32_naive(const Tensor &a, const Tensor &b, Tensor &c);

}

#endif // KL_MATMUL_CPU_FLOAT32_NAIVE_HPP