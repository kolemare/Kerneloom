#ifndef KL_BACKWARD_MSE_LOSS_CUDA_FLOAT32_CUH
#define KL_BACKWARD_MSE_LOSS_CUDA_FLOAT32_CUH

#include <cnn/losses/reduction.hpp>

#include <core/tensor.hpp>

namespace kl
{

    void backward_mse_loss_cuda_float32(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &grad_prediction,
        Reduction reduction);

}

#endif // KL_BACKWARD_MSE_LOSS_CUDA_FLOAT32_CUH