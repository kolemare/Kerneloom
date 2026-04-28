#ifndef KL_MATMUL_CPU_HPP
#define KL_MATMUL_CPU_HPP

#include <core/tensor.hpp>

namespace kl
{

    void matmul_cpu_float32(const Tensor &a, const Tensor &b, Tensor &c);

}

#endif // KL_MATMUL_CPU_HPP