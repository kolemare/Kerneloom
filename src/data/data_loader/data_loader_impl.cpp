#include "data/data_loader/data_loader_impl.hpp"

#include <data/image_decoder.hpp>

#include <core/dtype.hpp>
#include <core/memory_type.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kl
{

    namespace data_loader_internal
    {

        MemoryType host_memory_type(
            Device device,
            bool pin_host_memory)
        {
            if (!pin_host_memory)
            {
                return MemoryType::Default;
            }

            switch (device.type())
            {
            case DeviceType::CPU:
                return MemoryType::Default;

            case DeviceType::CUDA:
                return MemoryType::CudaPinnedHost;

            case DeviceType::ROCM:
                return MemoryType::RocmPinnedHost;

            default:
                return MemoryType::Default;
            }
        }

    }

    DataLoader::Impl::EpochState::EpochState(
        std::size_t generation,
        std::shared_ptr<
            const std::vector<std::size_t>>
            order,
        std::size_t total_batches)
        : generation(
              generation),
          order(
              std::move(order)),
          total_batches(
              total_batches)
    {
    }

    DataLoader::Impl::Impl(
        std::vector<ImageSample> samples,
        ImageTransform transform,
        Device device,
        DataLoaderOptions options,
        std::shared_ptr<DataLoaderMemoryBudget> memory_budget)
        : samples_(
              std::move(samples)),
          transform_(
              std::move(transform)),
          device_(
              device),
          options_(
              options),
          memory_budget_(
              std::move(memory_budget))
    {
        if (samples_.empty())
        {
            throw std::runtime_error(
                "DataLoader received an empty dataset");
        }

        if (options_.batch_size == 0)
        {
            throw std::runtime_error(
                "DataLoader batch size must be greater than zero");
        }

        if (options_.loader_workers == 0)
        {
            throw std::runtime_error(
                "DataLoader worker count must be greater than zero");
        }

        const std::size_t batch_bytes =
            options_.batch_size *
                transform_
                    .output_numel() *
                dtype_size(
                    options_
                        .input_dtype) +
            options_.batch_size *
                dtype_size(
                    DType::Int32);

        const bool uses_device =
            device_.type() !=
            DeviceType::CPU;

        if (memory_budget_ ==
            nullptr)
        {
            memory_budget_ =
                data_loader_internal::default_data_loader_memory_budget(
                    device_,
                    options_
                        .memory);
        }

        if (memory_budget_->device().type() !=
            device_.type())
        {
            throw std::runtime_error(
                "DataLoader memory budget device does not match DataLoader device");
        }

        if (options_
                .automatic_memory_planning)
        {
            memory_reservation_ =
                memory_budget_
                    ->reserve_auto(
                        batch_bytes,
                        options_
                            .loader_workers,
                        uses_device);

            has_memory_reservation_ =
                true;

            memory_plan_ =
                memory_reservation_
                    .plan;

            options_.decoded_cache_bytes =
                memory_plan_
                    .decoded_cache_bytes;

            options_.host_prefetch_batches =
                memory_plan_
                    .host_prefetch_batches;

            options_.device_prefetch_batches =
                memory_plan_
                    .device_prefetch_batches;
        }
        else
        {
            memory_plan_.batch_bytes =
                batch_bytes;

            memory_plan_.decoded_cache_bytes =
                options_
                    .decoded_cache_bytes;

            memory_plan_.host_prefetch_batches =
                options_
                    .host_prefetch_batches;

            memory_plan_.device_prefetch_batches =
                uses_device
                    ? options_
                          .device_prefetch_batches
                    : 0;

            memory_reservation_ =
                memory_budget_
                    ->reserve_manual(
                        batch_bytes,
                        memory_plan_
                            .decoded_cache_bytes,
                        memory_plan_
                            .host_prefetch_batches,
                        memory_plan_
                            .device_prefetch_batches,
                        options_
                            .loader_workers,
                        uses_device);

            has_memory_reservation_ =
                true;

            memory_plan_ =
                memory_reservation_
                    .plan;

            options_.decoded_cache_bytes =
                memory_plan_
                    .decoded_cache_bytes;

            options_.host_prefetch_batches =
                memory_plan_
                    .host_prefetch_batches;

            options_.device_prefetch_batches =
                memory_plan_
                    .device_prefetch_batches;
        }

        if (options_.host_prefetch_batches == 0)
        {
            throw std::runtime_error(
                "DataLoader host prefetch batch count must be greater than zero");
        }

        if (uses_device &&
            options_.device_prefetch_batches == 0)
        {
            throw std::runtime_error(
                "DataLoader device prefetch batch count must be greater than zero");
        }

        decoded_cache_ =
            std::make_shared<DecodedImageCache>(
                options_
                    .decoded_cache_bytes);

        host_queue_ =
            std::make_unique<
                BlockingQueue<PreparedBatch>>(
                options_
                    .host_prefetch_batches);

        host_batch_pool_ =
            std::make_shared<BatchStoragePool>(
                options_
                        .host_prefetch_batches +
                    options_
                        .loader_workers,
                transform_
                    .output_channels(),
                transform_
                    .output_height(),
                transform_
                    .output_width(),
                options_
                    .input_dtype,
                Device::cpu(),
                data_loader_internal::host_memory_type(
                    device_,
                    options_
                        .pin_host_memory));

        if (uses_device)
        {
            device_queue_ =
                std::make_unique<
                    BlockingQueue<PreparedBatch>>(
                    options_
                        .device_prefetch_batches);

            device_batch_pool_ =
                std::make_shared<BatchStoragePool>(
                    options_
                            .device_prefetch_batches +
                        1,
                    transform_
                        .output_channels(),
                    transform_
                        .output_height(),
                    transform_
                        .output_width(),
                    options_
                        .input_dtype,
                    device_,
                    MemoryType::Default);

            transfer_stream_ =
                std::make_unique<TransferStream>(
                    device_);
        }

        start_workers();

        if (uses_device)
        {
            start_transfer_worker();
        }

        reset_epoch();
    }

    DataLoader::Impl::~Impl()
    {
        stop_workers();

        if (has_memory_reservation_ &&
            memory_budget_ != nullptr)
        {
            memory_budget_
                ->release(
                    memory_reservation_);
        }
    }

    void DataLoader::Impl::reset_epoch()
    {
        rethrow_worker_exception();

        std::vector<std::size_t> order(
            samples_.size());

        std::iota(
            order.begin(),
            order.end(),
            0);

        if (options_.shuffle)
        {
            std::mt19937 generator(
                options_.seed +
                static_cast<std::uint32_t>(
                    epoch_));

            std::shuffle(
                order.begin(),
                order.end(),
                generator);
        }

        ++epoch_;

        const std::size_t generation =
            generation_
                .fetch_add(1) +
            1;

        total_batches_ =
            batch_count();

        next_batch_to_return_ =
            0;

        pending_host_batches_
            .clear();

        pending_device_batches_
            .clear();

        host_queue_->clear();

        if (device_queue_ !=
            nullptr)
        {
            device_queue_->clear();
        }

        std::shared_ptr<
            const std::vector<std::size_t>>
            shared_order =
                std::make_shared<
                    const std::vector<std::size_t>>(
                    std::move(order));

        {
            std::lock_guard<std::mutex> lock(
                epoch_mutex_);

            current_epoch_ =
                std::make_shared<EpochState>(
                    generation,
                    std::move(
                        shared_order),
                    total_batches_);
        }

        epoch_cv_.notify_all();
    }

    bool DataLoader::Impl::has_next() const
    {
        return next_batch_to_return_ <
               total_batches_;
    }

    Batch DataLoader::Impl::next()
    {
        rethrow_worker_exception();

        if (!has_next())
        {
            throw std::runtime_error(
                "DataLoader::next called without remaining batch");
        }

        if (device_.type() ==
            DeviceType::CPU)
        {
            return next_from_queue(
                *host_queue_,
                pending_host_batches_);
        }

        return next_from_queue(
            *device_queue_,
            pending_device_batches_);
    }

    std::size_t DataLoader::Impl::sample_count() const
    {
        return samples_.size();
    }

    std::size_t DataLoader::Impl::batch_count() const
    {
        if (options_.drop_last)
        {
            return samples_.size() /
                   options_.batch_size;
        }

        return (
                   samples_.size() +
                   options_.batch_size -
                   1) /
               options_.batch_size;
    }

    std::size_t DataLoader::Impl::host_prefetched_batch_count() const
    {
        return host_queue_->size();
    }

    std::size_t DataLoader::Impl::device_prefetched_batch_count() const
    {
        if (device_queue_ ==
            nullptr)
        {
            return 0;
        }

        return device_queue_->size();
    }

    std::size_t DataLoader::Impl::decoded_cache_image_count() const
    {
        return decoded_cache_
            ->image_count();
    }

    std::size_t DataLoader::Impl::decoded_cache_used_bytes() const
    {
        return decoded_cache_
            ->used_bytes();
    }

    std::uint64_t DataLoader::Impl::decoded_cache_hit_count() const
    {
        return decoded_cache_
            ->hit_count();
    }

    std::uint64_t DataLoader::Impl::decoded_cache_miss_count() const
    {
        return decoded_cache_
            ->miss_count();
    }

    std::size_t DataLoader::Impl::pooled_host_batch_count() const
    {
        return host_batch_pool_
            ->allocated_count();
    }

    std::size_t DataLoader::Impl::available_pooled_host_batch_count() const
    {
        return host_batch_pool_
            ->available_count();
    }

    std::size_t DataLoader::Impl::pooled_device_batch_count() const
    {
        if (device_batch_pool_ ==
            nullptr)
        {
            return 0;
        }

        return device_batch_pool_
            ->allocated_count();
    }

    std::size_t DataLoader::Impl::available_pooled_device_batch_count() const
    {
        if (device_batch_pool_ ==
            nullptr)
        {
            return 0;
        }

        return device_batch_pool_
            ->available_count();
    }

    const MemoryPlan &DataLoader::Impl::memory_plan() const
    {
        return memory_plan_;
    }

}