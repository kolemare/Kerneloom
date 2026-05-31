#include <data/data_loader.hpp>

#include <core/dtype.hpp>
#include <core/layout.hpp>
#include <core/shape.hpp>
#include <core/storage.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>

namespace kl
{

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
              options)
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

        reset_epoch();
    }

    DataLoader::~DataLoader()
    {
        stop_workers();
    }

    void DataLoader::reset_epoch()
    {
        stop_workers();

        order_.resize(
            samples_.size());

        std::iota(
            order_.begin(),
            order_.end(),
            0);

        if (options_.shuffle)
        {
            std::mt19937 generator(
                options_.seed +
                static_cast<std::uint32_t>(
                    epoch_));

            std::shuffle(
                order_.begin(),
                order_.end(),
                generator);
        }

        ++epoch_;

        total_batches_ =
            batch_count();

        next_batch_to_prepare_.store(
            0);

        next_batch_to_return_ =
            0;

        stop_requested_.store(
            false);

        pending_batches_.clear();

        {
            std::lock_guard<std::mutex> lock(
                exception_mutex_);

            worker_exception_ =
                nullptr;
        }

        host_queue_ =
            std::make_unique<
                BlockingQueue<PreparedBatch>>(
                options_
                    .host_prefetch_batches);

        start_workers();
    }

    bool DataLoader::has_next() const
    {
        return next_batch_to_return_ <
               total_batches_;
    }

    Batch DataLoader::next()
    {
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
            Batch batch =
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

            if (prepared->index ==
                next_batch_to_return_)
            {
                Batch batch =
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

    void DataLoader::start_workers()
    {
        const std::size_t worker_count =
            std::min(
                options_.loader_workers,
                std::max<std::size_t>(
                    1,
                    total_batches_));

        workers_.reserve(
            worker_count);

        for (std::size_t i = 0;
             i < worker_count;
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

        if (host_queue_ != nullptr)
        {
            host_queue_->close();
        }

        for (std::thread &worker :
             workers_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }

        workers_.clear();

        host_queue_.reset();
    }

    void DataLoader::worker_loop()
    {
        try
        {
            while (!stop_requested_.load())
            {
                const std::size_t batch_index =
                    next_batch_to_prepare_
                        .fetch_add(1);

                if (batch_index >=
                    total_batches_)
                {
                    return;
                }

                PreparedBatch prepared{
                    batch_index,
                    prepare_host_batch(
                        batch_index)};

                if (!host_queue_->push(
                        std::move(
                            prepared)))
                {
                    return;
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
        }
    }

    Batch DataLoader::prepare_host_batch(
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
            transform_.output_channels();

        const std::size_t height =
            transform_.output_height();

        const std::size_t width =
            transform_.output_width();

        Tensor inputs(
            Shape{
                current_batch_size,
                channels,
                height,
                width},
            options_.input_dtype,
            Device::cpu(),
            Layout::NCHW,
            Storage::RowMajor);

        Tensor targets(
            Shape{
                current_batch_size},
            DType::Int32,
            Device::cpu(),
            Layout::Unknown,
            Storage::RowMajor);

        std::byte *inputs_data =
            static_cast<std::byte *>(
                inputs.data());

        std::int32_t *targets_data =
            static_cast<std::int32_t *>(
                targets.data());

        const std::size_t image_bytes =
            transform_.output_numel() *
            dtype_size(
                options_.input_dtype);

        ImageDecoder decoder;

        for (std::size_t n = 0;
             n < current_batch_size;
             ++n)
        {
            const ImageSample &sample =
                samples_[order_[begin +
                                n]];

            const Image image =
                decoder.decode(
                    sample.path);

            transform_.write_chw(
                image,
                inputs_data +
                    n *
                        image_bytes,
                options_.input_dtype);

            targets_data[n] =
                sample.label;
        }

        return Batch{
            std::move(inputs),
            std::move(targets)};
    }

    Batch DataLoader::move_to_target_device(
        Batch batch) const
    {
        if (device_.type() ==
            DeviceType::CPU)
        {
            return batch;
        }

        return Batch{
            batch.inputs.to(
                device_),
            batch.targets.to(
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