#include <backend/backend.hpp>

namespace kl
{

    Device default_device()
    {
#if defined(KERNELOOM_DEFAULT_BACKEND_CUDA)
        return Device::cuda();
#elif defined(KERNELOOM_DEFAULT_BACKEND_ROCM)
        return Device::rocm();
#else
        return Device::cpu();
#endif
    }

}