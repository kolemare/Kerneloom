#ifndef KL_BINARY_CROSS_ENTROPY_CPU_FLOAT32_HPP
#define KL_BINARY_CROSS_ENTROPY_CPU_FLOAT32_HPP

#include <cnn/losses/reduction.hpp>

#include <core/tensor.hpp>

namespace kl
{

    void binary_cross_entropy_cpu_float32(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &result,
        Reduction reduction);

}

#endif // KL_BINARY_CROSS_ENTROPY_CPU_FLOAT32_HPP