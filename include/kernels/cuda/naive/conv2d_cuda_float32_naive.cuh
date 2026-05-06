#ifndef KL_CONV2D_CUDA_FLOAT32_NAIVE_CUH
#define KL_CONV2D_CUDA_FLOAT32_NAIVE_CUH

#include <core/tensor.hpp>

namespace kl
{

    void conv2d_cuda_float32_naive(
        const Tensor &input,
        const Tensor &kernels,
        Tensor &result);

}

#endif // KL_CONV2D_CUDA_FLOAT32_NAIVE_CUH