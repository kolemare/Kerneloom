#ifndef KL_POOLING2D_OPTIONS_HPP
#define KL_POOLING2D_OPTIONS_HPP

#include <cstddef>

namespace kl
{

    struct Pooling2dOptions
    {
        std::size_t kernel_h = 2;
        std::size_t kernel_w = 2;

        std::size_t stride_h = 2;
        std::size_t stride_w = 2;

        std::size_t padding_h = 0;
        std::size_t padding_w = 0;
    };

}

#endif // KL_POOLING2D_OPTIONS_HPP