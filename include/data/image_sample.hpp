#ifndef KL_IMAGE_SAMPLE_HPP
#define KL_IMAGE_SAMPLE_HPP

#include <cstdint>
#include <filesystem>

namespace kl
{

    struct ImageSample
    {
        std::filesystem::path path;
        std::int32_t label;
    };

}

#endif // KL_IMAGE_SAMPLE_HPP