#ifndef KL_BACKWARD_LINEAR_GRAD_INPUT_CUDA_FLOAT32_CUH
#define KL_BACKWARD_LINEAR_GRAD_INPUT_CUDA_FLOAT32_CUH

#include <core/tensor.hpp>

namespace kl
{

    void backward_linear_grad_input_cuda_float32(
        const Tensor &weights,
        const Tensor &grad_output,
        Tensor &grad_input);

}

#endif // KL_BACKWARD_LINEAR_GRAD_INPUT_CUDA_FLOAT32_CUH