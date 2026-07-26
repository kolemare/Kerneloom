#ifndef KL_CUDA_COPY_CUH
#define KL_CUDA_COPY_CUH

#include <core/tensor.hpp>

#include <cstddef>

namespace kl
{

    void cuda_copy(
        Tensor &dst,
        const Tensor &src);

    void cuda_copy_to_host(
        void *destination,
        const Tensor &source,
        std::size_t source_offset_bytes,
        std::size_t nbytes);

}

#endif // KL_CUDA_COPY_CUH