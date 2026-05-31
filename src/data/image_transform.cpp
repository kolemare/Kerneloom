#include <data/image_transform.hpp>

#include <cstddef>
#include <stdexcept>

namespace kl
{

    ImageTransform::ImageTransform(
        std::size_t output_width,
        std::size_t output_height)
        : output_width_(output_width),
          output_height_(output_height)
    {
        if (output_width_ == 0 ||
            output_height_ == 0)
        {
            throw std::runtime_error(
                "ImageTransform output dimensions must be greater than zero");
        }
    }

    void ImageTransform::write_chw(
        const Image &image,
        float *destination) const
    {
        if (destination == nullptr)
        {
            throw std::runtime_error(
                "ImageTransform received null destination");
        }

        if (image.channels() != 3)
        {
            throw std::runtime_error(
                "ImageTransform currently expects RGB images");
        }

        const std::uint8_t *source =
            image.data();

        const std::size_t plane_size =
            output_width_ *
            output_height_;

        for (std::size_t y = 0;
             y < output_height_;
             ++y)
        {
            const std::size_t source_y =
                y * image.height() /
                output_height_;

            for (std::size_t x = 0;
                 x < output_width_;
                 ++x)
            {
                const std::size_t source_x =
                    x * image.width() /
                    output_width_;

                const std::size_t source_index =
                    (source_y * image.width() +
                     source_x) *
                    image.channels();

                const std::size_t output_index =
                    y * output_width_ +
                    x;

                for (std::size_t channel = 0;
                     channel < 3;
                     ++channel)
                {
                    destination[channel * plane_size +
                                output_index] =
                        static_cast<float>(
                            source[source_index +
                                   channel]) /
                        255.0f;
                }
            }
        }
    }

    std::size_t ImageTransform::output_width() const
    {
        return output_width_;
    }

    std::size_t ImageTransform::output_height() const
    {
        return output_height_;
    }

    std::size_t ImageTransform::output_channels() const
    {
        return 3;
    }

    std::size_t ImageTransform::output_numel() const
    {
        return output_channels() *
               output_width_ *
               output_height_;
    }

}