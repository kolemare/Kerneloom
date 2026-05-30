#ifndef KL_SOFTMAX_CUDA_FLOAT32_CUH
#define KL_SOFTMAX_CUDA_FLOAT32_CUH

#include <core/tensor.hpp>

namespace kl
{

    void softmax_cuda_float32(
        Tensor &tensor);

}

#endif // KL_SOFTMAX_CUDA_FLOAT32_CUH