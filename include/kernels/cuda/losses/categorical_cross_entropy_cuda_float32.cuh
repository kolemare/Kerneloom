#ifndef KL_CATEGORICAL_CROSS_ENTROPY_CUDA_FLOAT32_CUH
#define KL_CATEGORICAL_CROSS_ENTROPY_CUDA_FLOAT32_CUH

#include <cnn/losses/reduction.hpp>

#include <core/tensor.hpp>

namespace kl
{

    void categorical_cross_entropy_cuda_float32(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &result,
        Reduction reduction);

}

#endif // KL_CATEGORICAL_CROSS_ENTROPY_CUDA_FLOAT32_CUH