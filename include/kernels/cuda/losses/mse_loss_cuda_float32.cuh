#ifndef KL_MSE_LOSS_CUDA_FLOAT32_CUH
#define KL_MSE_LOSS_CUDA_FLOAT32_CUH

#include <cnn/losses/reduction.hpp>

#include <core/tensor.hpp>

namespace kl
{

    void mse_loss_cuda_float32(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &result,
        Reduction reduction);

}

#endif // KL_MSE_LOSS_CUDA_FLOAT32_CUH