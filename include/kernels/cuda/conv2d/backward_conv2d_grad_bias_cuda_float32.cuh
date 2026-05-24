#ifndef KL_BACKWARD_CONV2D_GRAD_BIAS_CUDA_FLOAT32_CUH
#define KL_BACKWARD_CONV2D_GRAD_BIAS_CUDA_FLOAT32_CUH

#include <core/tensor.hpp>

namespace kl
{

    void backward_conv2d_grad_bias_cuda_float32(
        const Tensor &grad_output,
        Tensor &grad_bias);

}

#endif // KL_BACKWARD_CONV2D_GRAD_BIAS_CUDA_FLOAT32_CUH