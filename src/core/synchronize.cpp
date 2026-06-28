#include <core/synchronize.hpp>

#if defined(KL_ENABLE_CUDA)
#include <backend/cuda/cuda_synchronize.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <backend/rocm/rocm_synchronize.hiph>
#endif

#include <stdexcept>

namespace kl
{

    void synchronize(
        Device device)
    {
        switch (device.type())
        {
        case DeviceType::CPU:
            return;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            cuda_synchronize();
            return;
#else
            throw std::runtime_error(
                "CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            rocm_synchronize();
            return;
#else
            throw std::runtime_error(
                "ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error(
                "unknown DeviceType");
        }
    }

}