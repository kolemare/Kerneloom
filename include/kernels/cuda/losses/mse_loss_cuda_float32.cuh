#ifndef KL_MSE_LOSS_CUDA_FLOAT32_CUH
#define KL_MSE_LOSS_CUDA_FLOAT32_CUH

#include <cnn/losses/reduction.hpp>

#include <core/tensor.hpp>

#include <cstddef>

namespace kl
{

    void mse_loss_cuda_float32(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &result,
        Reduction reduction,
        std::size_t valid_sample_count);

}

#endif // KL_MSE_LOSS_CUDA_FLOAT32_CUH