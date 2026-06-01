#ifndef KL_MEMORY_PLAN_HPP
#define KL_MEMORY_PLAN_HPP

#include <cstddef>

namespace kl
{

    struct MemoryPlan
    {
        std::size_t available_ram_bytes =
            0;

        std::size_t available_vram_bytes =
            0;

        std::size_t batch_bytes =
            0;

        std::size_t decoded_cache_bytes =
            0;

        std::size_t host_prefetch_batches =
            0;

        std::size_t device_prefetch_batches =
            0;
    };

}

#endif // KL_MEMORY_PLAN_HPP