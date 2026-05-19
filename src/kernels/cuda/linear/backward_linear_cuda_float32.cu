#include <kernels/cuda/linear/backward_linear_cuda_float32.cuh>

#include <kernels/cuda/linear/backward_linear_grad_bias_cuda_float32.cuh>
#include <kernels/cuda/linear/backward_linear_grad_input_cuda_float32.cuh>
#include <kernels/cuda/linear/backward_linear_grad_weights_cuda_float32.cuh>

namespace kl
{

    void backward_linear_cuda_float32(
        const Tensor &input,
        const Tensor &weights,
        const Tensor &grad_output,
        Tensor &grad_input,
        Tensor &grad_weights,
        Tensor *grad_bias)
    {
        backward_linear_grad_input_cuda_float32(
            weights,
            grad_output,
            grad_input);

        backward_linear_grad_weights_cuda_float32(
            input,
            grad_output,
            grad_weights);

        if (grad_bias != nullptr)
        {
            backward_linear_grad_bias_cuda_float32(
                grad_output,
                *grad_bias);
        }
    }

}