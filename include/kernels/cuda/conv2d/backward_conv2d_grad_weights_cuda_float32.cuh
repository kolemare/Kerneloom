#ifndef KL_BACKWARD_CONV2D_GRAD_WEIGHTS_CUDA_FLOAT32_CUH
#define KL_BACKWARD_CONV2D_GRAD_WEIGHTS_CUDA_FLOAT32_CUH

#include <cnn/options/conv2d_options.hpp>

#include <core/tensor.hpp>

namespace kl
{

    void backward_conv2d_grad_weights_cuda_float32(
        const Tensor &input,
        const Tensor &grad_output,
        Tensor &grad_weights,
        const Conv2dOptions &options);

}

#endif // KL_BACKWARD_CONV2D_GRAD_WEIGHTS_CUDA_FLOAT32_CUH