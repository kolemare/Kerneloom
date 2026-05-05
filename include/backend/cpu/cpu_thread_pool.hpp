#ifndef KL_CPU_THREAD_POOL_HPP
#define KL_CPU_THREAD_POOL_HPP

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace kl
{

    class CpuThreadPool
    {
    public:
        explicit CpuThreadPool(std::size_t thread_count = std::thread::hardware_concurrency());
        ~CpuThreadPool();

        CpuThreadPool(const CpuThreadPool &) = delete;
        CpuThreadPool &operator=(const CpuThreadPool &) = delete;

        void enqueue(std::function<void()> task);
        void wait();

        std::size_t thread_count() const;

    private:
        void worker_loop();

    private:
        std::vector<std::thread> workers_;
        std::queue<std::function<void()>> tasks_;

        mutable std::mutex mutex_;
        std::condition_variable task_cv_;
        std::condition_variable done_cv_;

        std::size_t active_tasks_ = 0;
        bool stop_ = false;
    };

    CpuThreadPool &cpu_thread_pool();

}

#endif // KL_CPU_THREAD_POOL_HPP