#ifndef KL_DATA_LOADER_HPP
#define KL_DATA_LOADER_HPP

#include <data/batch.hpp>
#include <data/data_loader_options.hpp>
#include <data/image_sample.hpp>
#include <data/image_transform.hpp>
#include <data/internal/blocking_queue.hpp>

#include <core/device.hpp>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace kl
{

    class DataLoader
    {
    public:
        DataLoader(
            std::vector<ImageSample> samples,
            ImageTransform transform,
            Device device,
            DataLoaderOptions options = {});

        ~DataLoader();

        DataLoader(const DataLoader &) = delete;
        DataLoader &operator=(const DataLoader &) = delete;

        void reset_epoch();

        bool has_next() const;

        Batch next();

        std::size_t sample_count() const;
        std::size_t batch_count() const;

        std::size_t host_prefetched_batch_count() const;

    private:
        struct EpochState
        {
            EpochState(
                std::size_t generation,
                std::shared_ptr<
                    const std::vector<std::size_t>>
                    order,
                std::size_t total_batches);

            std::size_t generation;

            std::shared_ptr<
                const std::vector<std::size_t>>
                order;

            std::size_t total_batches;

            std::atomic<std::size_t>
                next_batch_to_prepare{0};
        };

        struct PreparedBatch
        {
            std::size_t generation;
            std::size_t index;

            Batch batch;
        };

        void start_workers();
        void stop_workers();

        void worker_loop();

        Batch prepare_host_batch(
            const std::vector<std::size_t> &order,
            std::size_t batch_index) const;

        Batch move_to_target_device(
            Batch batch) const;

        void store_worker_exception(
            std::exception_ptr exception);

        void rethrow_worker_exception() const;

    private:
        std::vector<ImageSample> samples_;

        ImageTransform transform_;

        Device device_;
        DataLoaderOptions options_;

        std::size_t epoch_ = 0;
        std::size_t generation_ = 0;

        std::size_t next_batch_to_return_ = 0;
        std::size_t total_batches_ = 0;

        std::atomic<bool>
            stop_requested_{false};

        std::unique_ptr<
            BlockingQueue<PreparedBatch>>
            host_queue_;

        std::vector<std::thread>
            workers_;

        std::map<std::size_t, Batch>
            pending_batches_;

        mutable std::mutex
            epoch_mutex_;

        std::condition_variable
            epoch_cv_;

        std::shared_ptr<EpochState>
            current_epoch_;

        mutable std::mutex
            exception_mutex_;

        std::exception_ptr
            worker_exception_;
    };

}

#endif // KL_DATA_LOADER_HPP