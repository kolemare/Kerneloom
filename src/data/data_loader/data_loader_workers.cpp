#include "data/data_loader/data_loader_impl.hpp"

#include <memory>
#include <optional>
#include <thread>
#include <utility>

namespace kl
{

    void DataLoader::Impl::start_workers()
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

    void DataLoader::Impl::start_transfer_worker()
    {
        transfer_worker_ =
            std::thread(
                [this]()
                {
                    transfer_worker_loop();
                });
    }

    void DataLoader::Impl::stop_workers()
    {
        stop_requested_.store(
            true);

        if (host_queue_ !=
            nullptr)
        {
            host_queue_->close();
        }

        if (device_queue_ !=
            nullptr)
        {
            device_queue_->close();
        }

        if (host_batch_pool_ !=
            nullptr)
        {
            host_batch_pool_->close();
        }

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

    void DataLoader::Impl::worker_loop()
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

            if (host_queue_ !=
                nullptr)
            {
                host_queue_->close();
            }

            if (device_queue_ !=
                nullptr)
            {
                device_queue_->close();
            }

            if (host_batch_pool_ !=
                nullptr)
            {
                host_batch_pool_->close();
            }

            if (device_batch_pool_ !=
                nullptr)
            {
                device_batch_pool_->close();
            }

            epoch_cv_.notify_all();
        }
    }

    void DataLoader::Impl::transfer_worker_loop()
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

            if (host_queue_ !=
                nullptr)
            {
                host_queue_->close();
            }

            if (device_queue_ !=
                nullptr)
            {
                device_queue_->close();
            }

            if (host_batch_pool_ !=
                nullptr)
            {
                host_batch_pool_->close();
            }

            if (device_batch_pool_ !=
                nullptr)
            {
                device_batch_pool_->close();
            }

            epoch_cv_.notify_all();
        }
    }

}