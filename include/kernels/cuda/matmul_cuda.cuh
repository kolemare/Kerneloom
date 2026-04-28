#ifndef KL_MATMUL_CUDA_CUH
#define KL_MATMUL_CUDA_CUH

#include <core/tensor.hpp>

namespace kl
{

    void matmul_cuda_float32(const Tensor &a, const Tensor &b, Tensor &c);

}

#endif // KL_MATMUL_CUDA_CUH