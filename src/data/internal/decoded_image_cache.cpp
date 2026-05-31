#include <data/internal/decoded_image_cache.hpp>

#include <stdexcept>
#include <utility>

namespace kl
{

    DecodedImageCache::DecodedImageCache(
        std::size_t capacity_bytes)
        : capacity_bytes_(
              capacity_bytes)
    {
    }

    std::shared_ptr<const Image>
    DecodedImageCache::find(
        const std::filesystem::path &path)
    {
        const std::string key =
            make_key(path);

        std::lock_guard<std::mutex> lock(
            mutex_);

        const auto iterator =
            entries_.find(key);

        if (iterator ==
            entries_.end())
        {
            ++miss_count_;
            return nullptr;
        }

        lru_.splice(
            lru_.begin(),
            lru_,
            iterator->second.lru_iterator);

        iterator->second.lru_iterator =
            lru_.begin();

        ++hit_count_;

        return iterator->second.image;
    }

    void DecodedImageCache::insert(
        const std::filesystem::path &path,
        std::shared_ptr<const Image> image)
    {
        if (image == nullptr)
        {
            throw std::runtime_error(
                "DecodedImageCache received null image");
        }

        const std::size_t bytes =
            image->size();

        if (capacity_bytes_ == 0 ||
            bytes > capacity_bytes_)
        {
            return;
        }

        const std::string key =
            make_key(path);

        std::lock_guard<std::mutex> lock(
            mutex_);

        const auto existing =
            entries_.find(key);

        if (existing !=
            entries_.end())
        {
            used_bytes_ -=
                existing->second.bytes;

            lru_.erase(
                existing->second.lru_iterator);

            entries_.erase(
                existing);
        }

        evict_until_fits(
            bytes);

        lru_.push_front(
            key);

        entries_.emplace(
            key,
            Entry{
                std::move(image),
                bytes,
                lru_.begin()});

        used_bytes_ +=
            bytes;
    }

    std::size_t
    DecodedImageCache::capacity_bytes() const
    {
        return capacity_bytes_;
    }

    std::size_t
    DecodedImageCache::used_bytes() const
    {
        std::lock_guard<std::mutex> lock(
            mutex_);

        return used_bytes_;
    }

    std::size_t
    DecodedImageCache::image_count() const
    {
        std::lock_guard<std::mutex> lock(
            mutex_);

        return entries_.size();
    }

    std::uint64_t
    DecodedImageCache::hit_count() const
    {
        std::lock_guard<std::mutex> lock(
            mutex_);

        return hit_count_;
    }

    std::uint64_t
    DecodedImageCache::miss_count() const
    {
        std::lock_guard<std::mutex> lock(
            mutex_);

        return miss_count_;
    }

    std::string
    DecodedImageCache::make_key(
        const std::filesystem::path &path)
    {
        return path
            .lexically_normal()
            .string();
    }

    void DecodedImageCache::evict_until_fits(
        std::size_t required_bytes)
    {
        while (!lru_.empty() &&
               used_bytes_ +
                       required_bytes >
                   capacity_bytes_)
        {
            const std::string key =
                lru_.back();

            const auto iterator =
                entries_.find(key);

            if (iterator !=
                entries_.end())
            {
                used_bytes_ -=
                    iterator->second.bytes;

                entries_.erase(
                    iterator);
            }

            lru_.pop_back();
        }
    }

}