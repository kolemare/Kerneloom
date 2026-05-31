#include <data/image_transform.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace kl
{

    namespace
    {

        template <typename T>
        void write_normalized_chw(
            const Image &image,
            T *destination,
            std::size_t output_width,
            std::size_t output_height)
        {
            const std::uint8_t *source =
                image.data();

            const std::size_t plane_size =
                output_width *
                output_height;

            for (std::size_t y = 0;
                 y < output_height;
                 ++y)
            {
                const std::size_t source_y =
                    y *
                    image.height() /
                    output_height;

                for (std::size_t x = 0;
                     x < output_width;
                     ++x)
                {
                    const std::size_t source_x =
                        x *
                        image.width() /
                        output_width;

                    const std::size_t source_index =
                        (source_y *
                             image.width() +
                         source_x) *
                        image.channels();

                    const std::size_t output_index =
                        y *
                            output_width +
                        x;

                    for (std::size_t channel = 0;
                         channel < 3;
                         ++channel)
                    {
                        destination[channel *
                                        plane_size +
                                    output_index] =
                            static_cast<T>(
                                static_cast<float>(
                                    source[source_index +
                                           channel]) /
                                255.0f);
                    }
                }
            }
        }

        void write_uint8_chw(
            const Image &image,
            std::uint8_t *destination,
            std::size_t output_width,
            std::size_t output_height)
        {
            const std::uint8_t *source =
                image.data();

            const std::size_t plane_size =
                output_width *
                output_height;

            for (std::size_t y = 0;
                 y < output_height;
                 ++y)
            {
                const std::size_t source_y =
                    y *
                    image.height() /
                    output_height;

                for (std::size_t x = 0;
                     x < output_width;
                     ++x)
                {
                    const std::size_t source_x =
                        x *
                        image.width() /
                        output_width;

                    const std::size_t source_index =
                        (source_y *
                             image.width() +
                         source_x) *
                        image.channels();

                    const std::size_t output_index =
                        y *
                            output_width +
                        x;

                    for (std::size_t channel = 0;
                         channel < 3;
                         ++channel)
                    {
                        destination[channel *
                                        plane_size +
                                    output_index] =
                            source[source_index +
                                   channel];
                    }
                }
            }
        }

    }

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
        void *destination,
        DType dtype) const
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

        switch (dtype)
        {
        case DType::Float32:
            write_normalized_chw(
                image,
                static_cast<float *>(
                    destination),
                output_width_,
                output_height_);
            return;

        case DType::Float64:
            write_normalized_chw(
                image,
                static_cast<double *>(
                    destination),
                output_width_,
                output_height_);
            return;

        case DType::UInt8:
            write_uint8_chw(
                image,
                static_cast<std::uint8_t *>(
                    destination),
                output_width_,
                output_height_);
            return;

        default:
            throw std::runtime_error(
                "ImageTransform does not support requested dtype yet");
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