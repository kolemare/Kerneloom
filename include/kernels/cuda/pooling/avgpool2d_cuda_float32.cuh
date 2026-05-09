#ifndef KL_AVGPOOL2D_CUDA_FLOAT32_CUH
#define KL_AVGPOOL2D_CUDA_FLOAT32_CUH

#include <cnn/options/pooling2d_options.hpp>
#include <core/tensor.hpp>

namespace kl
{

    void avgpool2d_cuda_float32(
        const Tensor &input,
        Tensor &result,
        const Pooling2dOptions &options);

}

#endif // KL_AVGPOOL2D_CUDA_FLOAT32_CUH