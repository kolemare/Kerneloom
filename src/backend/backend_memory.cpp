#include <backend/backend_memory.hpp>

#if defined(KL_ENABLE_CUDA)
#include <backend/cuda/cuda_memory.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <backend/rocm/rocm_memory.hiph>
#endif

#include <stdexcept>

namespace kl
{

    std::size_t available_device_memory_bytes(
        Device device)
    {
        switch (device.type())
        {
        case DeviceType::CPU:
            return 0;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            return available_cuda_memory_bytes();
#else
            throw std::runtime_error(
                "CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            return available_rocm_memory_bytes();
#else
            throw std::runtime_error(
                "ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error(
                "unknown DeviceType in available_device_memory_bytes");
        }
    }

}