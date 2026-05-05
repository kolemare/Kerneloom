#include <backend/cpu/cpu_parallel.hpp>

#include <backend/cpu/cpu_thread_pool.hpp>

#include <algorithm>

namespace kl
{

    void cpu_parallel_for(
        std::size_t begin,
        std::size_t end,
        const std::function<void(std::size_t begin, std::size_t end)> &function)
    {
        if (begin >= end)
        {
            return;
        }

        const std::size_t total_work = end - begin;

        CpuThreadPool &pool = cpu_thread_pool();

        const std::size_t worker_count = std::min<std::size_t>(
            pool.thread_count(),
            total_work);

        if (worker_count <= 1)
        {
            function(begin, end);
            return;
        }

        const std::size_t chunk_size = (total_work + worker_count - 1) / worker_count;

        for (std::size_t worker = 0; worker < worker_count; ++worker)
        {
            const std::size_t chunk_begin = begin + worker * chunk_size;
            const std::size_t chunk_end = std::min(end, chunk_begin + chunk_size);

            if (chunk_begin >= chunk_end)
            {
                break;
            }

            pool.enqueue(
                [chunk_begin, chunk_end, &function]()
                {
                    function(chunk_begin, chunk_end);
                });
        }

        pool.wait();
    }

}