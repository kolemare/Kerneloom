#include <backend/rocm/rocm_synchronize.hiph>

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

namespace kl
{

    void rocm_synchronize()
    {
        const hipError_t error =
            hipDeviceSynchronize();

        if (error != hipSuccess)
        {
            throw std::runtime_error(
                std::string("ROCm device synchronization failed: ") +
                hipGetErrorString(error));
        }
    }

}