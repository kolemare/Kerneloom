#ifndef KL_LAYOUT_HPP
#define KL_LAYOUT_HPP

#include <string_view>

namespace kl
{

    enum class Layout
    {
        Unknown,

        // 3D image/tensor layouts
        CHW,
        HWC,

        // 4D image batch layouts
        NCHW,
        NHWC,

        // 5D volume/video layouts
        NCDHW,
        NDHWC
    };

    std::string_view layout_name(Layout layout);

}

#endif // KL_LAYOUT_HPP