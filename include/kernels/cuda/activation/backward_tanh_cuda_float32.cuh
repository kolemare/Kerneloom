#ifndef KL_BACKWARD_TANH_CUDA_FLOAT32_CUH
#define KL_BACKWARD_TANH_CUDA_FLOAT32_CUH

#include <core/tensor.hpp>

namespace kl
{

    void backward_tanh_cuda_float32(
        const Tensor &activation_output,
        Tensor &grad);

}

#endif // KL_BACKWARD_TANH_CUDA_FLOAT32_CUH