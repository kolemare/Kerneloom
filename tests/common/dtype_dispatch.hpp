#ifndef KL_TEST_DTYPE_DISPATCH_HPP
#define KL_TEST_DTYPE_DISPATCH_HPP

#include <core/dtype.hpp>

#include <stdexcept>

namespace kl::test
{

    template <typename Fn>
    void dispatchFloatDType(
        DType dtype,
        Fn &&fn)
    {
        switch (dtype)
        {
        case DType::Float32:
            fn.template operator()<float>();
            break;

        case DType::Float64:
            fn.template operator()<double>();
            break;

        default:
            throw std::runtime_error(
                "Unsupported floating-point DType in test");
        }
    }

}

#endif // KL_TEST_DTYPE_DISPATCH_HPP