#ifndef KL_SIGMOID_CUDA_FLOAT32_CUH
#define KL_SIGMOID_CUDA_FLOAT32_CUH

#include <core/tensor.hpp>

namespace kl
{

    void sigmoid_cuda_float32(Tensor &tensor);

}

#endif // KL_SIGMOID_CUDA_FLOAT32_CUH