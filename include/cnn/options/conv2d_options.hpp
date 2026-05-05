#ifndef KL_CONV2D_OPTIONS_HPP
#define KL_CONV2D_OPTIONS_HPP

#include <cstddef>

namespace kl
{

    struct Conv2dOptions
    {
        std::size_t stride_h = 1;
        std::size_t stride_w = 1;

        std::size_t padding_h = 0;
        std::size_t padding_w = 0;

        std::size_t dilation_h = 1;
        std::size_t dilation_w = 1;

        bool use_bias = false;
    };

}

#endif // KL_CONV2D_OPTIONS_HPP