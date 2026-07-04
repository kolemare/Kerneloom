#include <data/data_loader.hpp>

#include <data/data_loader_memory_budget.hpp>
#include <data/image_decoder.hpp>
#include <data/memory_planner.hpp>

#include <core/dtype.hpp>
#include <core/layout.hpp>
#include <core/memory_type.hpp>
#include <core/shape.hpp>
#include <core/storage.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kl
{

    namespace
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

    DataLoader::EpochState::EpochState(
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

    DataLoader::DataLoader(
        std::vector<ImageSample> samples,
        ImageTransform transform,
        Device device,
        DataLoaderOptions options)
        : DataLoader(
              std::move(samples),
              std::move(transform),
              device,
              std::move(options),
              nullptr)
    {
    }

    DataLoader::DataLoader(
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

        if (options_
                .automatic_memory_planning)
        {
            if (memory_budget_ !=
                nullptr)
            {
                if (memory_budget_->device().type() !=
                    device_.type())
                {
                    throw std::runtime_error(
                        "DataLoader memory budget device does not match DataLoader device");
                }

                memory_reservation_ =
                    memory_budget_
                        ->reserve_auto(
                            batch_bytes,
                            options_
                                .loader_workers,
                            options_
                                .memory,
                            uses_device);

                has_memory_reservation_ =
                    true;

                memory_plan_ =
                    memory_reservation_
                        .plan;
            }
            else
            {
                memory_plan_ =
                    create_memory_plan(
                        device_,
                        batch_bytes,
                        options_
                            .memory);
            }

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

            if (memory_budget_ !=
                nullptr)
            {
                if (memory_budget_->device().type() !=
                    device_.type())
                {
                    throw std::runtime_error(
                        "DataLoader memory budget device does not match DataLoader device");
                }

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
            std::make_shared<
                DecodedImageCache>(
                options_
                    .decoded_cache_bytes);

        host_queue_ =
            std::make_unique<
                BlockingQueue<PreparedBatch>>(
                options_
                    .host_prefetch_batches);

        host_batch_pool_ =
            std::make_shared<
                BatchStoragePool>(
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
                host_memory_type(
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
                std::make_shared<
                    BatchStoragePool>(
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
                std::make_unique<
                    TransferStream>(
                    device_);
        }

        start_workers();

        if (uses_device)
        {
            start_transfer_worker();
        }

        reset_epoch();
    }

    DataLoader::~DataLoader()
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

    void DataLoader::reset_epoch()
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

    bool DataLoader::has_next() const
    {
        return next_batch_to_return_ <
               total_batches_;
    }

    Batch DataLoader::next()
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

    Batch DataLoader::next_from_queue(
        BlockingQueue<PreparedBatch> &queue,
        std::map<
            std::size_t,
            std::shared_ptr<BatchStorage>>
            &pending_batches)
    {
        const auto pending =
            pending_batches.find(
                next_batch_to_return_);

        if (pending !=
            pending_batches.end())
        {
            std::shared_ptr<BatchStorage>
                storage =
                    std::move(
                        pending->second);

            pending_batches.erase(
                pending);

            ++next_batch_to_return_;

            return Batch(
                std::move(storage));
        }

        while (true)
        {
            std::optional<PreparedBatch>
                prepared =
                    queue.pop();

            if (!prepared.has_value())
            {
                rethrow_worker_exception();

                throw std::runtime_error(
                    "DataLoader queue closed before expected batch was produced");
            }

            if (prepared->generation !=
                generation_.load())
            {
                continue;
            }

            if (prepared->index ==
                next_batch_to_return_)
            {
                std::shared_ptr<BatchStorage>
                    storage =
                        std::move(
                            prepared->storage);

                ++next_batch_to_return_;

                return Batch(
                    std::move(storage));
            }

            pending_batches.emplace(
                prepared->index,
                std::move(
                    prepared->storage));
        }
    }

    std::size_t DataLoader::sample_count() const
    {
        return samples_.size();
    }

    std::size_t DataLoader::batch_count() const
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

    std::size_t
    DataLoader::host_prefetched_batch_count() const
    {
        return host_queue_->size();
    }

    std::size_t
    DataLoader::device_prefetched_batch_count() const
    {
        if (device_queue_ ==
            nullptr)
        {
            return 0;
        }

        return device_queue_->size();
    }

    std::size_t
    DataLoader::decoded_cache_image_count() const
    {
        return decoded_cache_
            ->image_count();
    }

    std::size_t
    DataLoader::decoded_cache_used_bytes() const
    {
        return decoded_cache_
            ->used_bytes();
    }

    std::uint64_t
    DataLoader::decoded_cache_hit_count() const
    {
        return decoded_cache_
            ->hit_count();
    }

    std::uint64_t
    DataLoader::decoded_cache_miss_count() const
    {
        return decoded_cache_
            ->miss_count();
    }

    std::size_t
    DataLoader::pooled_host_batch_count() const
    {
        return host_batch_pool_
            ->allocated_count();
    }

    std::size_t
    DataLoader::available_pooled_host_batch_count() const
    {
        return host_batch_pool_
            ->available_count();
    }

    std::size_t
    DataLoader::pooled_device_batch_count() const
    {
        if (device_batch_pool_ ==
            nullptr)
        {
            return 0;
        }

        return device_batch_pool_
            ->allocated_count();
    }

    std::size_t
    DataLoader::available_pooled_device_batch_count() const
    {
        if (device_batch_pool_ ==
            nullptr)
        {
            return 0;
        }

        return device_batch_pool_
            ->available_count();
    }

    const MemoryPlan &
    DataLoader::memory_plan() const
    {
        return memory_plan_;
    }

    void DataLoader::start_workers()
    {
        workers_.reserve(
            options_.loader_workers);

        for (std::size_t i = 0;
             i < options_.loader_workers;
             ++i)
        {
            workers_.emplace_back(
                [this]()
                {
                    worker_loop();
                });
        }
    }

    void DataLoader::start_transfer_worker()
    {
        transfer_worker_ =
            std::thread(
                [this]()
                {
                    transfer_worker_loop();
                });
    }

    void DataLoader::stop_workers()
    {
        stop_requested_.store(
            true);

        host_queue_->close();

        if (device_queue_ !=
            nullptr)
        {
            device_queue_->close();
        }

        host_batch_pool_->close();

        if (device_batch_pool_ !=
            nullptr)
        {
            device_batch_pool_->close();
        }

        epoch_cv_.notify_all();

        for (std::thread &worker :
             workers_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }

        workers_.clear();

        if (transfer_worker_
                .joinable())
        {
            transfer_worker_
                .join();
        }
    }

    void DataLoader::worker_loop()
    {
        try
        {
            std::shared_ptr<EpochState>
                previous_epoch;

            while (!stop_requested_.load())
            {
                std::shared_ptr<EpochState>
                    epoch;

                {
                    std::unique_lock<std::mutex> lock(
                        epoch_mutex_);

                    epoch_cv_.wait(
                        lock,
                        [this, &previous_epoch]()
                        {
                            return stop_requested_
                                       .load() ||
                                   current_epoch_ !=
                                       previous_epoch;
                        });

                    if (stop_requested_.load())
                    {
                        return;
                    }

                    epoch =
                        current_epoch_;
                }

                previous_epoch =
                    epoch;

                while (!stop_requested_.load())
                {
                    const std::size_t batch_index =
                        epoch
                            ->next_batch_to_prepare
                            .fetch_add(1);

                    if (batch_index >=
                        epoch->total_batches)
                    {
                        break;
                    }

                    std::shared_ptr<BatchStorage>
                        storage =
                            prepare_host_batch(
                                *epoch->order,
                                batch_index);

                    {
                        std::lock_guard<std::mutex> lock(
                            epoch_mutex_);

                        if (current_epoch_ !=
                            epoch)
                        {
                            break;
                        }
                    }

                    PreparedBatch prepared{
                        epoch->generation,
                        batch_index,
                        std::move(storage)};

                    if (!host_queue_->push(
                            std::move(
                                prepared)))
                    {
                        return;
                    }
                }
            }
        }
        catch (...)
        {
            if (!stop_requested_.load())
            {
                store_worker_exception(
                    std::current_exception());
            }

            stop_requested_.store(
                true);

            host_queue_->close();

            if (device_queue_ !=
                nullptr)
            {
                device_queue_->close();
            }

            host_batch_pool_->close();

            if (device_batch_pool_ !=
                nullptr)
            {
                device_batch_pool_->close();
            }

            epoch_cv_.notify_all();
        }
    }

    void DataLoader::transfer_worker_loop()
    {
        try
        {
            while (!stop_requested_.load())
            {
                std::optional<PreparedBatch>
                    prepared =
                        host_queue_->pop();

                if (!prepared.has_value())
                {
                    return;
                }

                if (prepared->generation !=
                    generation_.load())
                {
                    continue;
                }

                const std::size_t batch_size =
                    prepared
                        ->storage
                        ->inputs
                        .shape()[0];

                std::shared_ptr<BatchStorage>
                    device_storage =
                        device_batch_pool_
                            ->acquire(
                                batch_size);

                transfer_stream_
                    ->copy_async(
                        device_storage
                            ->inputs,
                        prepared
                            ->storage
                            ->inputs);

                transfer_stream_
                    ->copy_async(
                        device_storage
                            ->targets,
                        prepared
                            ->storage
                            ->targets);

                transfer_stream_
                    ->synchronize();

                device_storage
                    ->valid_sample_count =
                    prepared
                        ->storage
                        ->valid_sample_count;

                if (prepared->generation !=
                    generation_.load())
                {
                    continue;
                }

                PreparedBatch ready{
                    prepared->generation,
                    prepared->index,
                    std::move(
                        device_storage)};

                if (!device_queue_->push(
                        std::move(
                            ready)))
                {
                    return;
                }
            }
        }
        catch (...)
        {
            if (!stop_requested_.load())
            {
                store_worker_exception(
                    std::current_exception());
            }

            stop_requested_.store(
                true);

            host_queue_->close();

            if (device_queue_ !=
                nullptr)
            {
                device_queue_->close();
            }

            host_batch_pool_->close();

            if (device_batch_pool_ !=
                nullptr)
            {
                device_batch_pool_->close();
            }

            epoch_cv_.notify_all();
        }
    }

    std::shared_ptr<BatchStorage>
    DataLoader::prepare_host_batch(
        const std::vector<std::size_t> &order,
        std::size_t batch_index) const
    {
        const std::size_t begin =
            batch_index *
            options_.batch_size;

        const std::size_t remaining =
            samples_.size() -
            begin;

        const std::size_t valid_sample_count =
            std::min(
                options_.batch_size,
                remaining);

        std::shared_ptr<BatchStorage>
            storage =
                host_batch_pool_
                    ->acquire(
                        options_
                            .batch_size);

        storage->valid_sample_count =
            valid_sample_count;

        if (valid_sample_count <
            options_.batch_size)
        {
            std::memset(
                storage
                    ->inputs
                    .data(),
                0,
                storage
                    ->inputs
                    .nbytes());

            std::memset(
                storage
                    ->targets
                    .data(),
                0,
                storage
                    ->targets
                    .nbytes());
        }

        std::byte *inputs_data =
            static_cast<std::byte *>(
                storage
                    ->inputs
                    .data());

        std::int32_t *targets_data =
            static_cast<std::int32_t *>(
                storage
                    ->targets
                    .data());

        const std::size_t image_bytes =
            transform_
                .output_numel() *
            dtype_size(
                options_
                    .input_dtype);

        ImageDecoder decoder;

        for (std::size_t n = 0;
             n < valid_sample_count;
             ++n)
        {
            const ImageSample &sample =
                samples_[order[begin +
                               n]];

            std::shared_ptr<const Image>
                image =
                    decoded_cache_
                        ->find(
                            sample.path);

            if (image ==
                nullptr)
            {
                image =
                    std::make_shared<Image>(
                        decoder.decode(
                            sample.path));

                decoded_cache_
                    ->insert(
                        sample.path,
                        image);
            }

            transform_.write_chw(
                *image,
                inputs_data +
                    n *
                        image_bytes,
                options_
                    .input_dtype);

            targets_data[n] =
                sample.label;
        }

        return storage;
    }

    void DataLoader::store_worker_exception(
        std::exception_ptr exception)
    {
        std::lock_guard<std::mutex> lock(
            exception_mutex_);

        if (worker_exception_ ==
            nullptr)
        {
            worker_exception_ =
                exception;
        }
    }

    void DataLoader::rethrow_worker_exception() const
    {
        std::lock_guard<std::mutex> lock(
            exception_mutex_);

        if (worker_exception_ !=
            nullptr)
        {
            std::rethrow_exception(
                worker_exception_);
        }
    }

}