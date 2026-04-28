#ifndef KL_CPU_ALLOCATOR_HPP
#define KL_CPU_ALLOCATOR_HPP

#include <core/allocator.hpp>

namespace kl
{

    class CpuAllocator final : public Allocator
    {
    public:
        void *allocate(std::size_t nbytes) override;
        void deallocate(void *ptr) override;
    };

}

#endif // KL_CPU_ALLOCATOR_HPP