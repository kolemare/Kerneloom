#ifndef KL_RELU_CUDA_FLOAT32_CUH
#define KL_RELU_CUDA_FLOAT32_CUH

#include <core/tensor.hpp>

namespace kl
{

    void relu_cuda_float32(Tensor &tensor);

}

#endif // KL_RELU_CUDA_FLOAT32_CUH