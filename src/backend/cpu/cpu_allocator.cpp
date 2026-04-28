#include <backend/cpu/cpu_allocator.hpp>

#include <cstdlib>
#include <new>

namespace kl
{

    void *CpuAllocator::allocate(std::size_t nbytes)
    {
        if (nbytes == 0)
        {
            return nullptr;
        }

        void *ptr = std::malloc(nbytes);

        if (ptr == nullptr)
        {
            throw std::bad_alloc();
        }

        return ptr;
    }

    void CpuAllocator::deallocate(void *ptr)
    {
        std::free(ptr);
    }

}