#ifndef KL_CUDA_TRANSFER_STREAM_CUH
#define KL_CUDA_TRANSFER_STREAM_CUH

#include <core/tensor.hpp>

namespace kl
{

    void *cuda_transfer_stream_create();

    void cuda_transfer_stream_destroy(
        void *handle) noexcept;

    void cuda_transfer_stream_copy_async(
        void *handle,
        Tensor &destination,
        const Tensor &source);

    void cuda_transfer_stream_synchronize(
        void *handle);

}

#endif // KL_CUDA_TRANSFER_STREAM_CUH