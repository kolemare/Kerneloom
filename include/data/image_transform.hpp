#ifndef KL_IMAGE_TRANSFORM_HPP
#define KL_IMAGE_TRANSFORM_HPP

#include <data/image.hpp>

#include <cstddef>

namespace kl
{

    class ImageTransform
    {
    public:
        ImageTransform(
            std::size_t output_width,
            std::size_t output_height);

        void write_chw(
            const Image &image,
            float *destination) const;

        std::size_t output_width() const;
        std::size_t output_height() const;
        std::size_t output_channels() const;
        std::size_t output_numel() const;

    private:
        std::size_t output_width_;
        std::size_t output_height_;
    };

}

#endif // KL_IMAGE_TRANSFORM_HPP