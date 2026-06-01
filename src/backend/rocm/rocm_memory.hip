#include <backend/rocm/rocm_memory.hiph>

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

namespace kl
{

    std::size_t available_rocm_memory_bytes()
    {
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;

        const hipError_t error =
            hipMemGetInfo(
                &free_bytes,
                &total_bytes);

        if (error != hipSuccess)
        {
            throw std::runtime_error(
                std::string(
                    "hipMemGetInfo failed: ") +
                hipGetErrorString(
                    error));
        }

        return free_bytes;
    }

}