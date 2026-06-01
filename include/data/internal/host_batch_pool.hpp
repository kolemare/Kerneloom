#ifndef KL_HOST_BATCH_POOL_HPP
#define KL_HOST_BATCH_POOL_HPP

#include <data/batch.hpp>

#include <core/dtype.hpp>
#include <core/memory_type.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace kl
{

    class HostBatchPool final
        : public std::enable_shared_from_this<
              HostBatchPool>
    {
    public:
        HostBatchPool(
            std::size_t channels,
            std::size_t height,
            std::size_t width,
            DType input_dtype,
            MemoryType memory_type);

        std::shared_ptr<Batch> acquire(
            std::size_t batch_size);

        std::size_t allocated_batch_count() const;
        std::size_t available_batch_count() const;

    private:
        std::unique_ptr<Batch> create_batch(
            std::size_t batch_size);

        void release(
            std::size_t batch_size,
            std::unique_ptr<Batch> batch);

    private:
        std::size_t channels_;
        std::size_t height_;
        std::size_t width_;

        DType input_dtype_;
        MemoryType memory_type_;

        std::size_t allocated_batch_count_ = 0;

        std::unordered_map<
            std::size_t,
            std::vector<std::unique_ptr<Batch>>>
            available_batches_;

        mutable std::mutex mutex_;
    };

}

#endif // KL_HOST_BATCH_POOL_HPP