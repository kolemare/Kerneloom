#ifndef KL_BACKWARD_CATEGORICAL_CROSS_ENTROPY_CUDA_FLOAT32_CUH
#define KL_BACKWARD_CATEGORICAL_CROSS_ENTROPY_CUDA_FLOAT32_CUH

#include <cnn/losses/reduction.hpp>

#include <core/tensor.hpp>

#include <cstddef>

namespace kl
{

    void backward_categorical_cross_entropy_cuda_float32(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &grad_prediction,
        Reduction reduction,
        std::size_t valid_sample_count);

}

#endif // KL_BACKWARD_CATEGORICAL_CROSS_ENTROPY_CUDA_FLOAT32_CUH