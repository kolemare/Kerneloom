#include <core/dtype.hpp>

#include <stdexcept>

namespace kl
{

    std::size_t dtype_size(DType dtype)
    {
        switch (dtype)
        {
        case DType::Float32:
            return 4;

        case DType::Float64:
            return 8;

        case DType::Float16:
            return 2;

        case DType::BFloat16:
            return 2;

        case DType::Int8:
            return 1;

        case DType::UInt8:
            return 1;

        default:
            throw std::invalid_argument("unknown DType");
        }
    }

    std::string_view dtype_name(DType dtype)
    {
        switch (dtype)
        {
        case DType::Float32:
            return "float32";

        case DType::Float64:
            return "float64";

        case DType::Float16:
            return "float16";

        case DType::BFloat16:
            return "bfloat16";

        case DType::Int8:
            return "int8";

        case DType::UInt8:
            return "uint8";

        default:
            throw std::invalid_argument("unknown DType");
        }
    }

    bool dtype_is_floating(DType dtype)
    {
        return dtype == DType::Float32 ||
               dtype == DType::Float64 ||
               dtype == DType::Float16 ||
               dtype == DType::BFloat16;
    }

    bool dtype_is_integer(DType dtype)
    {
        return dtype == DType::Int8 ||
               dtype == DType::UInt8;
    }

}