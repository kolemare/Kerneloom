#include <data/data_loader.hpp>

#include "data/data_loader/data_loader_impl.hpp"

#include <utility>

namespace kl
{

    DataLoader::DataLoader(
        std::vector<ImageSample> samples,
        ImageTransform transform,
        Device device,
        DataLoaderOptions options)
        : DataLoader(
              std::move(samples),
              std::move(transform),
              device,
              options,
              data_loader_internal::default_data_loader_memory_budget(
                  device,
                  options
                      .memory))
    {
    }

    DataLoader::DataLoader(
        std::vector<ImageSample> samples,
        ImageTransform transform,
        Device device,
        DataLoaderOptions options,
        std::shared_ptr<DataLoaderMemoryBudget> memory_budget)
        : impl_(
              std::make_unique<Impl>(
                  std::move(samples),
                  std::move(transform),
                  device,
                  std::move(options),
                  std::move(memory_budget)))
    {
    }

    DataLoader::~DataLoader() =
        default;

    DataLoader::DataLoader(
        DataLoader &&) noexcept =
        default;

    DataLoader &DataLoader::operator=(
        DataLoader &&) noexcept =
        default;

    void DataLoader::set_expected_loader_count(
        std::size_t expected_loader_count)
    {
        data_loader_internal::set_expected_loader_count(
            expected_loader_count);
    }

    std::size_t DataLoader::expected_loader_count()
    {
        return data_loader_internal::expected_loader_count();
    }

    void DataLoader::reset_epoch()
    {
        impl_->reset_epoch();
    }

    bool DataLoader::has_next() const
    {
        return impl_->has_next();
    }

    Batch DataLoader::next()
    {
        return impl_->next();
    }

    std::size_t DataLoader::sample_count() const
    {
        return impl_->sample_count();
    }

    std::size_t DataLoader::batch_count() const
    {
        return impl_->batch_count();
    }

    std::size_t DataLoader::host_prefetched_batch_count() const
    {
        return impl_->host_prefetched_batch_count();
    }

    std::size_t DataLoader::device_prefetched_batch_count() const
    {
        return impl_->device_prefetched_batch_count();
    }

    std::size_t DataLoader::decoded_cache_image_count() const
    {
        return impl_->decoded_cache_image_count();
    }

    std::size_t DataLoader::decoded_cache_used_bytes() const
    {
        return impl_->decoded_cache_used_bytes();
    }

    std::uint64_t DataLoader::decoded_cache_hit_count() const
    {
        return impl_->decoded_cache_hit_count();
    }

    std::uint64_t DataLoader::decoded_cache_miss_count() const
    {
        return impl_->decoded_cache_miss_count();
    }

    std::size_t DataLoader::pooled_host_batch_count() const
    {
        return impl_->pooled_host_batch_count();
    }

    std::size_t DataLoader::available_pooled_host_batch_count() const
    {
        return impl_->available_pooled_host_batch_count();
    }

    std::size_t DataLoader::pooled_device_batch_count() const
    {
        return impl_->pooled_device_batch_count();
    }

    std::size_t DataLoader::available_pooled_device_batch_count() const
    {
        return impl_->available_pooled_device_batch_count();
    }

    const MemoryPlan &DataLoader::memory_plan() const
    {
        return impl_->memory_plan();
    }

}