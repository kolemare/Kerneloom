#ifndef KL_MEMORY_POLICY_HPP
#define KL_MEMORY_POLICY_HPP

#include <cstddef>
#include <limits>

namespace kl
{

    struct MemoryPolicy
    {
        float decoded_cache_ram_fraction =
            0.40f;

        float host_prefetch_ram_fraction =
            0.10f;

        float device_prefetch_vram_fraction =
            0.50f;

        std::size_t max_host_prefetch_batches =
            std::numeric_limits<std::size_t>::max();

        std::size_t max_device_prefetch_batches =
            std::numeric_limits<std::size_t>::max();
    };

}

#endif // KL_MEMORY_POLICY_HPP