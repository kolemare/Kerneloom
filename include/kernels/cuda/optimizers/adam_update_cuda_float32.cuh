#ifndef KL_ADAM_UPDATE_CUDA_FLOAT32_CUH
#define KL_ADAM_UPDATE_CUDA_FLOAT32_CUH

#include <core/tensor.hpp>

namespace kl
{

    void adam_update_cuda_float32(
        Tensor &value,
        const Tensor &grad,
        Tensor &first_moment,
        Tensor &second_moment,
        float learning_rate,
        float beta1,
        float beta2,
        float epsilon,
        float beta1_power,
        float beta2_power);

}

#endif // KL_ADAM_UPDATE_CUDA_FLOAT32_CUH