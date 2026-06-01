#include <data/internal/host_batch_pool.hpp>

#include <core/device.hpp>
#include <core/layout.hpp>
#include <core/shape.hpp>
#include <core/storage.hpp>

#include <memory>
#include <utility>

namespace kl
{

    HostBatchPool::HostBatchPool(
        std::size_t channels,
        std::size_t height,
        std::size_t width,
        DType input_dtype,
        MemoryType memory_type)
        : channels_(
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
    }

    std::shared_ptr<Batch>
    HostBatchPool::acquire(
        std::size_t batch_size)
    {
        std::unique_ptr<Batch>
            batch;

        {
            std::lock_guard<std::mutex> lock(
                mutex_);

            std::vector<std::unique_ptr<Batch>>
                &available =
                    available_batches_[batch_size];

            if (!available.empty())
            {
                batch =
                    std::move(
                        available.back());

                available.pop_back();
            }
            else
            {
                batch =
                    create_batch(
                        batch_size);

                ++allocated_batch_count_;
            }
        }

        std::weak_ptr<HostBatchPool>
            weak_pool =
                shared_from_this();

        return std::shared_ptr<Batch>(
            batch.release(),
            [weak_pool, batch_size](
                Batch *value)
            {
                std::unique_ptr<Batch>
                    batch(
                        value);

                const std::shared_ptr<
                    HostBatchPool>
                    pool =
                        weak_pool.lock();

                if (pool != nullptr)
                {
                    pool->release(
                        batch_size,
                        std::move(batch));
                }
            });
    }

    std::size_t
    HostBatchPool::allocated_batch_count() const
    {
        std::lock_guard<std::mutex> lock(
            mutex_);

        return allocated_batch_count_;
    }

    std::size_t
    HostBatchPool::available_batch_count() const
    {
        std::lock_guard<std::mutex> lock(
            mutex_);

        std::size_t count =
            0;

        for (const auto &entry :
             available_batches_)
        {
            count +=
                entry.second.size();
        }

        return count;
    }

    std::unique_ptr<Batch>
    HostBatchPool::create_batch(
        std::size_t batch_size)
    {
        return std::make_unique<Batch>(
            Batch{
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

    void HostBatchPool::release(
        std::size_t batch_size,
        std::unique_ptr<Batch> batch)
    {
        std::lock_guard<std::mutex> lock(
            mutex_);

        available_batches_[batch_size]
            .push_back(
                std::move(batch));
    }

}