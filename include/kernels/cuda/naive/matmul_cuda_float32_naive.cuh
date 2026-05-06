#ifndef KL_MATMUL_CUDA_FLOAT32_NAIVE_CUH
#define KL_MATMUL_CUDA_FLOAT32_NAIVE_CUH

#include <core/tensor.hpp>

namespace kl
{

    void matmul_cuda_float32_naive(const Tensor &a, const Tensor &b, Tensor &c);

}

#endif // KL_MATMUL_CUDA_FLOAT32_NAIVE_CUH