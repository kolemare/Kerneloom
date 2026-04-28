#include <core/layout.hpp>

#include <stdexcept>

namespace kl
{

    std::string_view layout_name(Layout layout)
    {
        switch (layout)
        {
        case Layout::Unknown:
            return "unknown";

        case Layout::CHW:
            return "chw";

        case Layout::HWC:
            return "hwc";

        case Layout::NCHW:
            return "nchw";

        case Layout::NHWC:
            return "nhwc";

        case Layout::NCDHW:
            return "ncdhw";

        case Layout::NDHWC:
            return "ndhwc";

        default:
            throw std::invalid_argument("unknown Layout");
        }
    }

}