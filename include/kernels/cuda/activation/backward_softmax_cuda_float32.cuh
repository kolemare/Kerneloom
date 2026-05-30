#ifndef KL_BACKWARD_SOFTMAX_CUDA_FLOAT32_CUH
#define KL_BACKWARD_SOFTMAX_CUDA_FLOAT32_CUH

#include <core/tensor.hpp>

namespace kl
{

    void backward_softmax_cuda_float32(
        const Tensor &activation_output,
        Tensor &grad);

}

#endif // KL_BACKWARD_SOFTMAX_CUDA_FLOAT32_CUH