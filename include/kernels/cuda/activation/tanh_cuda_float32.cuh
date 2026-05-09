#ifndef KL_TANH_CUDA_FLOAT32_CUH
#define KL_TANH_CUDA_FLOAT32_CUH

#include <core/tensor.hpp>

namespace kl
{

    void tanh_cuda_float32(Tensor &tensor);

}

#endif // KL_TANH_CUDA_FLOAT32_CUH