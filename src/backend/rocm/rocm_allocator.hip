#include <backend/rocm/rocm_allocator.hiph>

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

namespace kl
{

    namespace
    {

        void check_hip(hipError_t error, const char *message)
        {
            if (error != hipSuccess)
            {
                throw std::runtime_error(
                    std::string(message) + ": " + hipGetErrorString(error));
            }
        }

    }

    void *RocmAllocator::allocate(std::size_t nbytes)
    {
        if (nbytes == 0)
        {
            return nullptr;
        }

        void *ptr = nullptr;
        check_hip(hipMalloc(&ptr, nbytes), "hipMalloc failed");

        return ptr;
    }

    void RocmAllocator::deallocate(
        void *ptr) noexcept
    {
        if (ptr != nullptr)
        {
            static_cast<void>(
                hipFree(ptr));
        }
    }

}