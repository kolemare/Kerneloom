#ifndef KL_BATCH_STORAGE_POOL_HPP
#define KL_BATCH_STORAGE_POOL_HPP

#include <data/internal/batch_storage.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/memory_type.hpp>

#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace kl
{

    class BatchStoragePool final
        : public std::enable_shared_from_this<
              BatchStoragePool>
    {
    public:
        BatchStoragePool(
            std::size_t capacity,
            std::size_t channels,
            std::size_t height,
            std::size_t width,
            DType input_dtype,
            Device device,
            MemoryType memory_type =
                MemoryType::Default);

        std::shared_ptr<BatchStorage> acquire(
            std::size_t batch_size);

        void close();

        std::size_t capacity() const;
        std::size_t allocated_count() const;
        std::size_t available_count() const;

    private:
        std::unique_ptr<BatchStorage>
        create_storage(
            std::size_t batch_size) const;

        void release(
            std::size_t batch_size,
            std::unique_ptr<BatchStorage>
                storage) noexcept;

        bool has_available_storage() const;

    private:
        std::size_t capacity_;

        std::size_t channels_;
        std::size_t height_;
        std::size_t width_;

        DType input_dtype_;

        Device device_;
        MemoryType memory_type_;

        std::size_t allocated_count_ = 0;

        bool closed_ = false;

        std::unordered_map<
            std::size_t,
            std::vector<
                std::unique_ptr<BatchStorage>>>
            available_;

        mutable std::mutex mutex_;

        std::condition_variable
            condition_;
    };

}

#endif // KL_BATCH_STORAGE_POOL_HPP