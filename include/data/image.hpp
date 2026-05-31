#ifndef KL_IMAGE_HPP
#define KL_IMAGE_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kl
{

    class Image
    {
    public:
        Image(
            std::size_t width,
            std::size_t height,
            std::size_t channels,
            std::vector<std::uint8_t> pixels);

        std::size_t width() const;
        std::size_t height() const;
        std::size_t channels() const;
        std::size_t size() const;

        const std::uint8_t *data() const;

    private:
        std::size_t width_;
        std::size_t height_;
        std::size_t channels_;

        std::vector<std::uint8_t> pixels_;
    };

}

#endif // KL_IMAGE_HPP