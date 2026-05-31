#include <data/image_decoder.hpp>

#include <stb/stb_image.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace kl
{

    Image ImageDecoder::decode(
        const std::filesystem::path &path) const
    {
        int width = 0;
        int height = 0;
        int original_channels = 0;

        constexpr int requested_channels = 3;

        unsigned char *raw =
            stbi_load(
                path.string().c_str(),
                &width,
                &height,
                &original_channels,
                requested_channels);

        if (raw == nullptr)
        {
            throw std::runtime_error(
                "failed to decode image: " +
                path.string());
        }

        const std::size_t pixel_count =
            static_cast<std::size_t>(width) *
            static_cast<std::size_t>(height) *
            requested_channels;

        std::vector<std::uint8_t> pixels(
            raw,
            raw + pixel_count);

        stbi_image_free(raw);

        return Image(
            static_cast<std::size_t>(width),
            static_cast<std::size_t>(height),
            requested_channels,
            std::move(pixels));
    }

}