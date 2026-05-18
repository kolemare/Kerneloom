#ifndef KL_CONV2D_CUDA_FLOAT32_DISPATCHER_CUH
#define KL_CONV2D_CUDA_FLOAT32_DISPATCHER_CUH

#include <cnn/options/conv2d_options.hpp>
#include <core/tensor.hpp>

namespace kl
{

    void conv2d_cuda_float32_dispatcher(
        const Tensor &input,
        const Tensor &kernels,
        const Tensor *bias,
        Tensor &result,
        const Conv2dOptions &options);

}

#endif // KL_CONV2D_CUDA_FLOAT32_DISPATCHER_CUH