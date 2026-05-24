#ifndef KL_SGD_UPDATE_CUDA_FLOAT32_CUH
#define KL_SGD_UPDATE_CUDA_FLOAT32_CUH

#include <core/tensor.hpp>

namespace kl
{

    void sgd_update_cuda_float32(
        Tensor &value,
        const Tensor &grad,
        float learning_rate);

}

#endif // KL_SGD_UPDATE_CUDA_FLOAT32_CUH