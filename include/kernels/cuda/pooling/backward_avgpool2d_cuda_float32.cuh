#ifndef KL_BACKWARD_AVGPOOL2D_CUDA_FLOAT32_CUH
#define KL_BACKWARD_AVGPOOL2D_CUDA_FLOAT32_CUH

#include <cnn/options/pooling2d_options.hpp>

#include <core/tensor.hpp>

namespace kl
{

    void backward_avgpool2d_cuda_float32(
        const Tensor &grad_output,
        Tensor &grad_input,
        const Pooling2dOptions &options);

}

#endif // KL_BACKWARD_AVGPOOL2D_CUDA_FLOAT32_CUH