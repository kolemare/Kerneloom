#include <kernels/cuda/conv2d/backward_conv2d_cuda_float32.cuh>

#include <kernels/cuda/conv2d/backward_conv2d_grad_bias_cuda_float32.cuh>
#include <kernels/cuda/conv2d/backward_conv2d_grad_input_cuda_float32.cuh>
#include <kernels/cuda/conv2d/backward_conv2d_grad_weights_cuda_float32.cuh>

namespace kl
{

    void backward_conv2d_cuda_float32(
        const Tensor &input,
        const Tensor &weights,
        const Tensor &grad_output,
        Tensor &grad_input,
        Tensor &grad_weights,
        Tensor *grad_bias,
        const Conv2dOptions &options)
    {
        backward_conv2d_grad_input_cuda_float32(
            weights,
            grad_output,
            grad_input,
            options);

        backward_conv2d_grad_weights_cuda_float32(
            input,
            grad_output,
            grad_weights,
            options);

        if (grad_bias != nullptr)
        {
            backward_conv2d_grad_bias_cuda_float32(
                grad_output,
                *grad_bias);
        }
    }

}