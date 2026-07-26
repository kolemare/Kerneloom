#ifndef KL_COPY_HPP
#define KL_COPY_HPP

#include <core/tensor.hpp>

#include <cstddef>

namespace kl
{

    void copy(
        Tensor &dst,
        const Tensor &src);

    void copy_to_host(
        void *destination,
        const Tensor &source,
        std::size_t source_offset_bytes,
        std::size_t nbytes);

}

#endif // KL_COPY_HPP