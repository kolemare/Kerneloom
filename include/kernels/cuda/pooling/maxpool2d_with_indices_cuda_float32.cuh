#ifndef KL_MAXPOOL2D_WITH_INDICES_CUDA_FLOAT32_CUH
#define KL_MAXPOOL2D_WITH_INDICES_CUDA_FLOAT32_CUH

#include <cnn/options/pooling2d_options.hpp>

#include <core/tensor.hpp>

namespace kl
{

    void maxpool2d_with_indices_cuda_float32(
        const Tensor &input,
        Tensor &result,
        Tensor &indices,
        const Pooling2dOptions &options);

}

#endif // KL_MAXPOOL2D_WITH_INDICES_CUDA_FLOAT32_CUH