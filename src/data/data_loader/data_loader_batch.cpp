#include "data/data_loader/data_loader_impl.hpp"

#include <data/image_decoder.hpp>

#include <core/dtype.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace kl
{

    Batch DataLoader::Impl::next_from_queue(
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

    std::shared_ptr<BatchStorage>
    DataLoader::Impl::prepare_host_batch(
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

    void DataLoader::Impl::store_worker_exception(
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

    void DataLoader::Impl::rethrow_worker_exception() const
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