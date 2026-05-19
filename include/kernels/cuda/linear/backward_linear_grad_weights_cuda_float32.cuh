#ifndef KL_BACKWARD_LINEAR_GRAD_WEIGHTS_CUDA_FLOAT32_CUH
#define KL_BACKWARD_LINEAR_GRAD_WEIGHTS_CUDA_FLOAT32_CUH

#include <core/tensor.hpp>

namespace kl
{

    void backward_linear_grad_weights_cuda_float32(
        const Tensor &input,
        const Tensor &grad_output,
        Tensor &grad_weights);

}

#endif // KL_BACKWARD_LINEAR_GRAD_WEIGHTS_CUDA_FLOAT32_CUH