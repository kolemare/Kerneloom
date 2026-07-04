#ifndef KL_DATA_LOADER_HPP
#define KL_DATA_LOADER_HPP

#include <data/batch.hpp>
#include <data/data_loader/data_loader_memory_budget.hpp>
#include <data/data_loader/data_loader_options.hpp>
#include <data/image_sample.hpp>
#include <data/image_transform.hpp>
#include <data/memory_plan.hpp>

#include <core/device.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace kl
{

    class DataLoader
    {
    public:
        static void set_expected_loader_count(
            std::size_t expected_loader_count);

        static std::size_t expected_loader_count();

        DataLoader(
            std::vector<ImageSample> samples,
            ImageTransform transform,
            Device device,
            DataLoaderOptions options = {});

        DataLoader(
            std::vector<ImageSample> samples,
            ImageTransform transform,
            Device device,
            DataLoaderOptions options,
            std::shared_ptr<DataLoaderMemoryBudget> memory_budget);

        ~DataLoader();

        DataLoader(const DataLoader &) =
            delete;

        DataLoader &operator=(
            const DataLoader &) =
            delete;

        DataLoader(
            DataLoader &&) noexcept;

        DataLoader &operator=(
            DataLoader &&) noexcept;

        void reset_epoch();

        bool has_next() const;

        Batch next();

        std::size_t sample_count() const;

        std::size_t batch_count() const;

        std::size_t host_prefetched_batch_count() const;

        std::size_t device_prefetched_batch_count() const;

        std::size_t decoded_cache_image_count() const;

        std::size_t decoded_cache_used_bytes() const;

        std::uint64_t decoded_cache_hit_count() const;

        std::uint64_t decoded_cache_miss_count() const;

        std::size_t pooled_host_batch_count() const;

        std::size_t available_pooled_host_batch_count() const;

        std::size_t pooled_device_batch_count() const;

        std::size_t available_pooled_device_batch_count() const;

        const MemoryPlan &memory_plan() const;

    private:
        class Impl;

        std::unique_ptr<Impl> impl_;
    };

}

#endif // KL_DATA_LOADER_HPP