#ifndef KL_DATA_LOADER_HPP
#define KL_DATA_LOADER_HPP

#include <data/batch.hpp>
#include <data/data_loader_memory_budget.hpp>
#include <data/data_loader_options.hpp>
#include <data/image_sample.hpp>
#include <data/image_transform.hpp>
#include <data/internal/batch_storage.hpp>
#include <data/internal/batch_storage_pool.hpp>
#include <data/internal/blocking_queue.hpp>
#include <data/internal/decoded_image_cache.hpp>
#include <data/memory_plan.hpp>

#include <core/device.hpp>
#include <core/transfer_stream.hpp>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
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
            std::shared_ptr<BatchStorage> storage;
        };

        void start_workers();

        void stop_workers();

        void worker_loop();

        void start_transfer_worker();

        void transfer_worker_loop();

        Batch next_from_queue(
            BlockingQueue<PreparedBatch> &queue,
            std::map<
                std::size_t,
                std::shared_ptr<BatchStorage>>
                &pending_batches);

        std::shared_ptr<BatchStorage>
        prepare_host_batch(
            const std::vector<std::size_t> &order,
            std::size_t batch_index) const;

        void store_worker_exception(
            std::exception_ptr exception);

        void rethrow_worker_exception() const;

    private:
        std::vector<ImageSample> samples_;

        ImageTransform transform_;

        Device device_;

        DataLoaderOptions options_;

        MemoryPlan memory_plan_;

        std::size_t epoch_ =
            0;

        std::atomic<std::size_t> generation_{
            0};

        std::size_t next_batch_to_return_ =
            0;

        std::size_t total_batches_ =
            0;

        std::atomic<bool> stop_requested_{
            false};

        std::unique_ptr<
            BlockingQueue<PreparedBatch>>
            host_queue_;

        std::unique_ptr<
            BlockingQueue<PreparedBatch>>
            device_queue_;

        std::vector<std::thread> workers_;

        std::thread transfer_worker_;

        std::map<
            std::size_t,
            std::shared_ptr<BatchStorage>>
            pending_host_batches_;

        std::map<
            std::size_t,
            std::shared_ptr<BatchStorage>>
            pending_device_batches_;

        std::shared_ptr<BatchStoragePool>
            host_batch_pool_;

        std::shared_ptr<BatchStoragePool>
            device_batch_pool_;

        std::unique_ptr<TransferStream>
            transfer_stream_;

        mutable std::mutex epoch_mutex_;

        std::condition_variable epoch_cv_;

        std::shared_ptr<EpochState>
            current_epoch_;

        std::shared_ptr<DecodedImageCache>
            decoded_cache_;

        std::shared_ptr<DataLoaderMemoryBudget>
            memory_budget_;

        DataLoaderMemoryReservation
            memory_reservation_;

        bool has_memory_reservation_ =
            false;

        mutable std::mutex exception_mutex_;

        std::exception_ptr worker_exception_;
    };

}

#endif // KL_DATA_LOADER_HPP