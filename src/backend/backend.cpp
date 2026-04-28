#include <backend/backend.hpp>

namespace kl
{

        Device default_device()
        {
#if defined(KL_ENABLE_CUDA)
                return Device::cuda();
#elif defined(KL_ENABLE_ROCM)
                return Device::rocm();
#else
                return Device::cpu();
#endif
        }

}