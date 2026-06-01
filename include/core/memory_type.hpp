#ifndef KL_MEMORY_TYPE_HPP
#define KL_MEMORY_TYPE_HPP

namespace kl
{

    enum class MemoryType
    {
        Default,
        CudaPinnedHost,
        RocmPinnedHost
    };

}

#endif // KL_MEMORY_TYPE_HPP