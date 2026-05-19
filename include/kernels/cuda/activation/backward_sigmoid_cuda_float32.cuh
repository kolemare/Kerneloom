#ifndef KL_BACKWARD_SIGMOID_CUDA_FLOAT32_CUH
#define KL_BACKWARD_SIGMOID_CUDA_FLOAT32_CUH

#include <core/tensor.hpp>

namespace kl
{

    void backward_sigmoid_cuda_float32(
        const Tensor &activation_output,
        Tensor &grad);

}

#endif // KL_BACKWARD_SIGMOID_CUDA_FLOAT32_CUH