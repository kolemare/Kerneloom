#ifndef KL_LINEAR_CUDA_FLOAT32_CUH
#define KL_LINEAR_CUDA_FLOAT32_CUH

#include <core/tensor.hpp>

namespace kl
{

    void linear_cuda_float32(
        const Tensor &input,
        const Tensor &weights,
        const Tensor *bias,
        Tensor &result);

}

#endif // KL_LINEAR_CUDA_FLOAT32_CUH