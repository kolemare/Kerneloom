#ifndef KL_DATA_LOADER_OPTIONS_HPP
#define KL_DATA_LOADER_OPTIONS_HPP

#include <data/memory_policy.hpp>

#include <cstddef>
#include <cstdint>

namespace kl
{

    struct DataLoaderOptions
    {
        std::size_t batch_size = 1;

        bool shuffle = false;
        bool drop_last = false;

        std::uint32_t seed = 1337;

        MemoryPolicy memory;
    };

}

#endif // KL_DATA_LOADER_OPTIONS_HPP