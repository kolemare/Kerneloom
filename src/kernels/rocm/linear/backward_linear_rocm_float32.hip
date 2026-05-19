#include <kernels/rocm/linear/backward_linear_rocm_float32.hiph>

#include <kernels/rocm/linear/backward_linear_grad_bias_rocm_float32.hiph>
#include <kernels/rocm/linear/backward_linear_grad_input_rocm_float32.hiph>
#include <kernels/rocm/linear/backward_linear_grad_weights_rocm_float32.hiph>

namespace kl
{

    void backward_linear_rocm_float32(
        const Tensor &input,
        const Tensor &weights,
        const Tensor &grad_output,
        Tensor &grad_input,
        Tensor &grad_weights,
        Tensor *grad_bias)
    {
        backward_linear_grad_input_rocm_float32(
            weights,
            grad_output,
            grad_input);

        backward_linear_grad_weights_rocm_float32(
            input,
            grad_output,
            grad_weights);

        if (grad_bias != nullptr)
        {
            backward_linear_grad_bias_rocm_float32(
                grad_output,
                *grad_bias);
        }
    }

}