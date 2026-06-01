#include <data/data_loader.hpp>

#include <data/image_decoder.hpp>

#include <core/dtype.hpp>
#include <core/layout.hpp>
#include <core/memory_type.hpp>
#include <core/shape.hpp>
#include <core/storage.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
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
        : samples_(
              std::move(samples)),
          transform_(
              std::move(transform)),
          device_(
              device),
          options_(
              options),
          decoded_cache_(
              std::make_shared<
                  DecodedImageCache>(
                  options.decoded_cache_bytes))
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

        if (options_.host_prefetch_batches == 0)
        {
            throw std::runtime_error(
                "DataLoader host prefetch batch count must be greater than zero");
        }

        host_queue_ =
            std::make_unique<
                BlockingQueue<PreparedBatch>>(
                options_
                    .host_prefetch_batches);

        if (device_.type() !=
            DeviceType::CPU)
        {
            host_batch_pool_ =
                std::make_shared<
                    HostBatchPool>(
                    transform_
                        .output_channels(),
                    transform_
                        .output_height(),
                    transform_
                        .output_width(),
                    options_
                        .input_dtype,
                    host_memory_type(
                        device_,
                        options_
                            .pin_host_memory));
        }

        start_workers();
        reset_epoch();
    }

    DataLoader::~DataLoader()
    {
        stop_workers();
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
        ++generation_;

        total_batches_ =
            batch_count();

        next_batch_to_return_ =
            0;

        pending_batches_.clear();

        host_queue_->clear();

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
                    generation_,
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

        const auto pending =
            pending_batches_.find(
                next_batch_to_return_);

        if (pending !=
            pending_batches_.end())
        {
            std::shared_ptr<Batch>
                batch =
                    std::move(
                        pending->second);

            pending_batches_.erase(
                pending);

            ++next_batch_to_return_;

            return move_to_target_device(
                std::move(batch));
        }

        while (true)
        {
            std::optional<PreparedBatch>
                prepared =
                    host_queue_->pop();

            if (!prepared.has_value())
            {
                rethrow_worker_exception();

                throw std::runtime_error(
                    "DataLoader host queue closed before expected batch was produced");
            }

            if (prepared->generation !=
                generation_)
            {
                continue;
            }

            if (prepared->index ==
                next_batch_to_return_)
            {
                std::shared_ptr<Batch>
                    batch =
                        std::move(
                            prepared->batch);

                ++next_batch_to_return_;

                return move_to_target_device(
                    std::move(batch));
            }

            pending_batches_.emplace(
                prepared->index,
                std::move(
                    prepared->batch));
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
        if (host_batch_pool_ ==
            nullptr)
        {
            return 0;
        }

        return host_batch_pool_
            ->allocated_batch_count();
    }

    std::size_t
    DataLoader::available_pooled_host_batch_count() const
    {
        if (host_batch_pool_ ==
            nullptr)
        {
            return 0;
        }

        return host_batch_pool_
            ->available_batch_count();
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

    void DataLoader::stop_workers()
    {
        stop_requested_.store(
            true);

        host_queue_->close();

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
                            return stop_requested_.load() ||
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

                    std::shared_ptr<Batch>
                        batch =
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
                        std::move(batch)};

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
            store_worker_exception(
                std::current_exception());

            stop_requested_.store(
                true);

            host_queue_->close();
            epoch_cv_.notify_all();
        }
    }

    std::shared_ptr<Batch>
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

        const std::size_t current_batch_size =
            std::min(
                options_.batch_size,
                remaining);

        const std::size_t channels =
            transform_
                .output_channels();

        const std::size_t height =
            transform_
                .output_height();

        const std::size_t width =
            transform_
                .output_width();

        std::shared_ptr<Batch>
            batch;

        if (host_batch_pool_ !=
            nullptr)
        {
            batch =
                host_batch_pool_
                    ->acquire(
                        current_batch_size);
        }
        else
        {
            batch =
                std::make_shared<Batch>(
                    Batch{
                        Tensor(
                            Shape{
                                current_batch_size,
                                channels,
                                height,
                                width},
                            options_
                                .input_dtype,
                            Device::cpu(),
                            Layout::NCHW,
                            Storage::RowMajor),

                        Tensor(
                            Shape{
                                current_batch_size},
                            DType::Int32,
                            Device::cpu(),
                            Layout::Unknown,
                            Storage::RowMajor)});
        }

        std::byte *inputs_data =
            static_cast<std::byte *>(
                batch
                    ->inputs
                    .data());

        std::int32_t *targets_data =
            static_cast<std::int32_t *>(
                batch
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
             n < current_batch_size;
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

        return batch;
    }

    Batch DataLoader::move_to_target_device(
        std::shared_ptr<Batch> batch) const
    {
        if (device_.type() ==
            DeviceType::CPU)
        {
            return std::move(
                *batch);
        }

        return Batch{
            batch
                ->inputs
                .to(
                    device_),

            batch
                ->targets
                .to(
                    device_)};
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