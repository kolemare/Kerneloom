#ifndef KL_MEMORY_POLICY_HPP
#define KL_MEMORY_POLICY_HPP

#include <cstddef>

namespace kl
{

    struct MemoryPolicy
    {
        float usable_vram_fraction = 0.30f;
        float usable_pinned_ram_fraction = 0.05f;
        float decoded_cache_ram_fraction = 0.40f;

        std::size_t min_device_prefetch_batches = 2;
        std::size_t max_device_prefetch_batches = 4;

        std::size_t min_host_prefetch_batches = 4;
        std::size_t max_host_prefetch_batches = 16;
    };

}

#endif // KL_MEMORY_POLICY_HPP