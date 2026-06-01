#include <data/internal/batch_storage_pool.hpp>

#include <core/device.hpp>
#include <core/layout.hpp>
#include <core/shape.hpp>
#include <core/storage.hpp>

#include <stdexcept>
#include <utility>

namespace kl
{

    BatchStoragePool::BatchStoragePool(
        std::size_t capacity,
        std::size_t channels,
        std::size_t height,
        std::size_t width,
        DType input_dtype,
        MemoryType memory_type)
        : capacity_(
              capacity),
          channels_(
              channels),
          height_(
              height),
          width_(
              width),
          input_dtype_(
              input_dtype),
          memory_type_(
              memory_type)
    {
        if (capacity_ == 0)
        {
            throw std::runtime_error(
                "BatchStoragePool capacity must be greater than zero");
        }

        if (channels_ == 0 ||
            height_ == 0 ||
            width_ == 0)
        {
            throw std::runtime_error(
                "BatchStoragePool dimensions must be greater than zero");
        }
    }

    std::shared_ptr<BatchStorage>
    BatchStoragePool::acquire(
        std::size_t batch_size)
    {
        if (batch_size == 0)
        {
            throw std::runtime_error(
                "BatchStoragePool batch size must be greater than zero");
        }

        std::unique_ptr<BatchStorage>
            storage;

        {
            std::unique_lock<std::mutex> lock(
                mutex_);

            condition_.wait(
                lock,
                [this, batch_size]()
                {
                    const auto iterator =
                        available_.find(
                            batch_size);

                    const bool matching_available =
                        iterator !=
                            available_.end() &&
                        !iterator
                             ->second
                             .empty();

                    return closed_ ||
                           matching_available ||
                           allocated_count_ <
                               capacity_ ||
                           has_available_storage();
                });

            if (closed_)
            {
                throw std::runtime_error(
                    "BatchStoragePool is closed");
            }

            std::vector<
                std::unique_ptr<BatchStorage>>
                &matching =
                    available_[batch_size];

            if (!matching.empty())
            {
                storage =
                    std::move(
                        matching.back());

                matching.pop_back();
            }
            else
            {
                if (allocated_count_ >=
                    capacity_)
                {
                    for (auto &entry :
                         available_)
                    {
                        if (entry.second.empty())
                        {
                            continue;
                        }

                        entry.second.pop_back();

                        --allocated_count_;

                        break;
                    }
                }

                storage =
                    create_storage(
                        batch_size);

                ++allocated_count_;
            }
        }

        std::weak_ptr<BatchStoragePool>
            weak_pool =
                shared_from_this();

        return std::shared_ptr<BatchStorage>(
            storage.release(),
            [weak_pool, batch_size](
                BatchStorage *value)
            {
                std::unique_ptr<BatchStorage>
                    storage(
                        value);

                const std::shared_ptr<
                    BatchStoragePool>
                    pool =
                        weak_pool.lock();

                if (pool != nullptr)
                {
                    pool->release(
                        batch_size,
                        std::move(storage));
                }
            });
    }

    void BatchStoragePool::close()
    {
        {
            std::lock_guard<std::mutex> lock(
                mutex_);

            if (closed_)
            {
                return;
            }

            closed_ = true;

            std::size_t available_count =
                0;

            for (const auto &entry :
                 available_)
            {
                available_count +=
                    entry.second.size();
            }

            available_.clear();

            allocated_count_ -=
                available_count;
        }

        condition_.notify_all();
    }

    std::size_t
    BatchStoragePool::capacity() const
    {
        return capacity_;
    }

    std::size_t
    BatchStoragePool::allocated_count() const
    {
        std::lock_guard<std::mutex> lock(
            mutex_);

        return allocated_count_;
    }

    std::size_t
    BatchStoragePool::available_count() const
    {
        std::lock_guard<std::mutex> lock(
            mutex_);

        std::size_t count =
            0;

        for (const auto &entry :
             available_)
        {
            count +=
                entry.second.size();
        }

        return count;
    }

    std::unique_ptr<BatchStorage>
    BatchStoragePool::create_storage(
        std::size_t batch_size) const
    {
        return std::make_unique<BatchStorage>(
            BatchStorage{
                Tensor(
                    Shape{
                        batch_size,
                        channels_,
                        height_,
                        width_},
                    input_dtype_,
                    Device::cpu(),
                    Layout::NCHW,
                    Storage::RowMajor,
                    memory_type_),

                Tensor(
                    Shape{
                        batch_size},
                    DType::Int32,
                    Device::cpu(),
                    Layout::Unknown,
                    Storage::RowMajor,
                    memory_type_)});
    }

    void BatchStoragePool::release(
        std::size_t batch_size,
        std::unique_ptr<BatchStorage> storage) noexcept
    {
        if (storage == nullptr)
        {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(
                mutex_);

            if (closed_)
            {
                if (allocated_count_ > 0)
                {
                    --allocated_count_;
                }

                return;
            }

            available_[batch_size]
                .push_back(
                    std::move(storage));
        }

        condition_.notify_one();
    }

    bool
    BatchStoragePool::has_available_storage() const
    {
        for (const auto &entry :
             available_)
        {
            if (!entry.second.empty())
            {
                return true;
            }
        }

        return false;
    }

}