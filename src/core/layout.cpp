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

        case Layout::NCHW:
            return "nchw";

        default:
            throw std::invalid_argument("unknown Layout");
        }
    }

}