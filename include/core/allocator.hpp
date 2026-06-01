#ifndef KL_ALLOCATOR_HPP
#define KL_ALLOCATOR_HPP

#include <core/device.hpp>
#include <core/memory_type.hpp>

#include <cstddef>

namespace kl
{

    class Allocator
    {
    public:
        virtual ~Allocator() = default;

        virtual void *allocate(
            std::size_t nbytes) = 0;

        virtual void deallocate(
            void *ptr) = 0;
    };

    Allocator &allocator_for(
        Device device,
        MemoryType memory_type =
            MemoryType::Default);

}

#endif // KL_ALLOCATOR_HPP