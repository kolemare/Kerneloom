#ifndef KL_DECODED_IMAGE_CACHE_HPP
#define KL_DECODED_IMAGE_CACHE_HPP

#include <data/image.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace kl
{

    class DecodedImageCache
    {
    public:
        explicit DecodedImageCache(
            std::size_t capacity_bytes);

        std::shared_ptr<const Image> find(
            const std::filesystem::path &path);

        void insert(
            const std::filesystem::path &path,
            std::shared_ptr<const Image> image);

        std::size_t capacity_bytes() const;
        std::size_t used_bytes() const;
        std::size_t image_count() const;

        std::uint64_t hit_count() const;
        std::uint64_t miss_count() const;

    private:
        struct Entry
        {
            std::shared_ptr<const Image> image;
            std::size_t bytes;

            std::list<std::string>::iterator
                lru_iterator;
        };

        static std::string make_key(
            const std::filesystem::path &path);

        void evict_until_fits(
            std::size_t required_bytes);

    private:
        std::size_t capacity_bytes_;
        std::size_t used_bytes_ = 0;

        std::list<std::string> lru_;

        std::unordered_map<
            std::string,
            Entry>
            entries_;

        std::uint64_t hit_count_ = 0;
        std::uint64_t miss_count_ = 0;

        mutable std::mutex mutex_;
    };

}

#endif // KL_DECODED_IMAGE_CACHE_HPP