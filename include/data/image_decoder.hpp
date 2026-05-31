#ifndef KL_IMAGE_DECODER_HPP
#define KL_IMAGE_DECODER_HPP

#include <data/image.hpp>

#include <filesystem>

namespace kl
{

    class ImageDecoder
    {
    public:
        Image decode(
            const std::filesystem::path &path) const;
    };

}

#endif // KL_IMAGE_DECODER_HPP