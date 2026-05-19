#ifndef KL_BACKWARD_LINEAR_CUDA_FLOAT32_CUH
#define KL_BACKWARD_LINEAR_CUDA_FLOAT32_CUH

#include <core/tensor.hpp>

namespace kl
{

    void backward_linear_cuda_float32(
        const Tensor &input,
        const Tensor &weights,
        const Tensor &grad_output,
        Tensor &grad_input,
        Tensor &grad_weights,
        Tensor *grad_bias);

}

#endif // KL_BACKWARD_LINEAR_CUDA_FLOAT32_CUH