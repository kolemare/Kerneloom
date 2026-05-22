#ifndef KL_DTYPE_HPP
#define KL_DTYPE_HPP

#include <cstddef>
#include <string_view>

namespace kl
{

    enum class DType
    {
        Float32,
        Float64,
        Float16,
        BFloat16,

        Int32,
        Int8,
        UInt8
    };

    std::size_t dtype_size(DType dtype);
    std::string_view dtype_name(DType dtype);

    bool dtype_is_floating(DType dtype);
    bool dtype_is_integer(DType dtype);

}

#endif // KL_DTYPE_HPP