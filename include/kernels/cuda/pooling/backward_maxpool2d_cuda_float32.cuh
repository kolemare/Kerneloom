#ifndef KL_BACKWARD_MAXPOOL2D_CUDA_FLOAT32_CUH
#define KL_BACKWARD_MAXPOOL2D_CUDA_FLOAT32_CUH

#include <core/tensor.hpp>

namespace kl
{

    void backward_maxpool2d_cuda_float32(
        const Tensor &indices,
        const Tensor &grad_output,
        Tensor &grad_input);

}

#endif // KL_BACKWARD_MAXPOOL2D_CUDA_FLOAT32_CUH