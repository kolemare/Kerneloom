#include <backend/cpu/cpu_thread_pool.hpp>

#include <algorithm>
#include <stdexcept>

namespace kl
{

    CpuThreadPool::CpuThreadPool(std::size_t thread_count)
    {
        if (thread_count == 0)
        {
            thread_count = 1;
        }

        workers_.reserve(thread_count);

        for (std::size_t i = 0; i < thread_count; ++i)
        {
            workers_.emplace_back(
                [this]()
                {
                    worker_loop();
                });
        }
    }

    CpuThreadPool::~CpuThreadPool()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }

        task_cv_.notify_all();

        for (std::thread &worker : workers_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
    }

    void CpuThreadPool::enqueue(std::function<void()> task)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);

            if (stop_)
            {
                throw std::runtime_error("cannot enqueue task on stopped CpuThreadPool");
            }

            tasks_.push(std::move(task));
        }

        task_cv_.notify_one();
    }

    void CpuThreadPool::wait()
    {
        std::unique_lock<std::mutex> lock(mutex_);

        done_cv_.wait(
            lock,
            [this]()
            {
                return tasks_.empty() && active_tasks_ == 0;
            });
    }

    std::size_t CpuThreadPool::thread_count() const
    {
        return workers_.size();
    }

    void CpuThreadPool::worker_loop()
    {
        while (true)
        {
            std::function<void()> task;

            {
                std::unique_lock<std::mutex> lock(mutex_);

                task_cv_.wait(
                    lock,
                    [this]()
                    {
                        return stop_ || !tasks_.empty();
                    });

                if (stop_ && tasks_.empty())
                {
                    return;
                }

                task = std::move(tasks_.front());
                tasks_.pop();
                ++active_tasks_;
            }

            task();

            {
                std::lock_guard<std::mutex> lock(mutex_);
                --active_tasks_;

                if (tasks_.empty() && active_tasks_ == 0)
                {
                    done_cv_.notify_all();
                }
            }
        }
    }

    CpuThreadPool &cpu_thread_pool()
    {
        static CpuThreadPool pool;
        return pool;
    }

}