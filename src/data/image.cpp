#include <data/image.hpp>

#include <stdexcept>
#include <utility>

namespace kl
{

    Image::Image(
        std::size_t width,
        std::size_t height,
        std::size_t channels,
        std::vector<std::uint8_t> pixels)
        : width_(width),
          height_(height),
          channels_(channels),
          pixels_(std::move(pixels))
    {
        if (width_ == 0 ||
            height_ == 0 ||
            channels_ == 0)
        {
            throw std::runtime_error(
                "Image dimensions must be greater than zero");
        }

        if (pixels_.size() !=
            width_ * height_ * channels_)
        {
            throw std::runtime_error(
                "Image pixel count does not match dimensions");
        }
    }

    std::size_t Image::width() const
    {
        return width_;
    }

    std::size_t Image::height() const
    {
        return height_;
    }

    std::size_t Image::channels() const
    {
        return channels_;
    }

    std::size_t Image::size() const
    {
        return pixels_.size();
    }

    const std::uint8_t *Image::data() const
    {
        return pixels_.data();
    }

}