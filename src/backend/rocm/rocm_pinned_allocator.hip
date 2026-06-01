#include <backend/rocm/rocm_pinned_allocator.hiph>

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

namespace kl
{

    namespace
    {

        void check_hip(
            hipError_t error,
            const char *message)
        {
            if (error != hipSuccess)
            {
                throw std::runtime_error(
                    std::string(message) +
                    ": " +
                    hipGetErrorString(error));
            }
        }

    }

    void *RocmPinnedAllocator::allocate(
        std::size_t nbytes)
    {
        if (nbytes == 0)
        {
            return nullptr;
        }

        void *ptr =
            nullptr;

        check_hip(
            hipHostMalloc(
                &ptr,
                nbytes,
                hipHostMallocDefault),
            "hipHostMalloc failed");

        return ptr;
    }

    void RocmPinnedAllocator::deallocate(
        void *ptr)
    {
        if (ptr == nullptr)
        {
            return;
        }

        check_hip(
            hipHostFree(
                ptr),
            "hipHostFree failed");
    }

}