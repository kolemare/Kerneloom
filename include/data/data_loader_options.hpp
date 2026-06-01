#ifndef KL_DATA_LOADER_OPTIONS_HPP
#define KL_DATA_LOADER_OPTIONS_HPP

#include <data/memory_policy.hpp>

#include <core/dtype.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <thread>

namespace kl
{

    inline std::size_t
    default_loader_worker_count()
    {
        const std::size_t count =
            std::thread::hardware_concurrency();

        return std::max<std::size_t>(
            1,
            count /
                2);
    }

    struct DataLoaderOptions
    {
        std::size_t batch_size = 1;

        DType input_dtype =
            DType::Float32;

        bool shuffle = false;
        bool drop_last = false;

        std::uint32_t seed = 1337;

        std::size_t loader_workers =
            default_loader_worker_count();

        std::size_t host_prefetch_batches =
            4;

        std::size_t device_prefetch_batches =
            3;

        bool pin_host_memory =
            true;

        std::size_t decoded_cache_bytes =
            0;

        MemoryPolicy memory;
    };

}

#endif // KL_DATA_LOADER_OPTIONS_HPP