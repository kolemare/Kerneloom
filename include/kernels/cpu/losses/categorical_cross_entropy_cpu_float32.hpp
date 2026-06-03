#ifndef KL_CATEGORICAL_CROSS_ENTROPY_CPU_FLOAT32_HPP
#define KL_CATEGORICAL_CROSS_ENTROPY_CPU_FLOAT32_HPP

#include <cnn/losses/reduction.hpp>

#include <core/tensor.hpp>

#include <cstddef>

namespace kl
{

    void categorical_cross_entropy_cpu_float32(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &result,
        Reduction reduction,
        std::size_t valid_sample_count);

}

#endif // KL_CATEGORICAL_CROSS_ENTROPY_CPU_FLOAT32_HPP