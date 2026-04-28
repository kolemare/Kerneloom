#ifndef KL_CUDA_COPY_CUH
#define KL_CUDA_COPY_CUH

#include <core/tensor.hpp>

namespace kl
{

    void cuda_copy(Tensor &dst, const Tensor &src);

}

#endif // KL_CUDA_COPY_CUH