#ifndef KL_BLOCKING_QUEUE_HPP
#define KL_BLOCKING_QUEUE_HPP

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace kl
{

    template <typename T>
    class BlockingQueue
    {
    public:
        explicit BlockingQueue(
            std::size_t capacity)
            : capacity_(capacity)
        {
            if (capacity_ == 0)
            {
                throw std::runtime_error(
                    "BlockingQueue capacity must be greater than zero");
            }
        }

        BlockingQueue(const BlockingQueue &) = delete;
        BlockingQueue &operator=(const BlockingQueue &) = delete;

        bool push(
            T value)
        {
            std::unique_lock<std::mutex> lock(
                mutex_);

            push_cv_.wait(
                lock,
                [this]()
                {
                    return closed_ ||
                           queue_.size() <
                               capacity_;
                });

            if (closed_)
            {
                return false;
            }

            queue_.push_back(
                std::move(value));

            pop_cv_.notify_one();

            return true;
        }

        std::optional<T> pop()
        {
            std::unique_lock<std::mutex> lock(
                mutex_);

            pop_cv_.wait(
                lock,
                [this]()
                {
                    return closed_ ||
                           !queue_.empty();
                });

            if (queue_.empty())
            {
                return std::nullopt;
            }

            T value =
                std::move(
                    queue_.front());

            queue_.pop_front();

            push_cv_.notify_one();

            return value;
        }

        void close()
        {
            {
                std::lock_guard<std::mutex> lock(
                    mutex_);

                closed_ = true;
            }

            push_cv_.notify_all();
            pop_cv_.notify_all();
        }

    private:
        std::size_t capacity_;

        std::deque<T> queue_;

        bool closed_ = false;

        std::mutex mutex_;
        std::condition_variable push_cv_;
        std::condition_variable pop_cv_;
    };

}

#endif // KL_BLOCKING_QUEUE_HPP