#include <kernels/rocm/conv2d/backward_conv2d_rocm_float32.hiph>

#include <kernels/rocm/conv2d/backward_conv2d_grad_bias_rocm_float32.hiph>
#include <kernels/rocm/conv2d/backward_conv2d_grad_input_rocm_float32.hiph>
#include <kernels/rocm/conv2d/backward_conv2d_grad_weights_rocm_float32.hiph>

namespace kl
{

    void backward_conv2d_rocm_float32(
        const Tensor &input,
        const Tensor &weights,
        const Tensor &grad_output,
        Tensor &grad_input,
        Tensor &grad_weights,
        Tensor *grad_bias,
        const Conv2dOptions &options)
    {
        backward_conv2d_grad_input_rocm_float32(
            weights,
            grad_output,
            grad_input,
            options);

        backward_conv2d_grad_weights_rocm_float32(
            input,
            grad_output,
            grad_weights,
            options);

        if (grad_bias != nullptr)
        {
            backward_conv2d_grad_bias_rocm_float32(
                grad_output,
                *grad_bias);
        }
    }

}