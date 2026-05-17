#ifndef KL_LAYOUT_HPP
#define KL_LAYOUT_HPP

#include <string_view>

namespace kl
{

    enum class Layout
    {
        Unknown,
        NCHW
    };

    std::string_view layout_name(Layout layout);

}

#endif // KL_LAYOUT_HPP