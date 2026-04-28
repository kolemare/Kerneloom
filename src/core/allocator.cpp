#include <core/allocator.hpp>

#include <backend/cpu/cpu_allocator.hpp>

#if defined(KL_ENABLE_CUDA)
#include <backend/cuda/cuda_allocator.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <backend/rocm/rocm_allocator.hiph>
#endif

#include <stdexcept>

namespace kl
{

        Allocator &allocator_for(Device device)
        {
                static CpuAllocator cpu_allocator;

#if defined(KL_ENABLE_CUDA)
                static CudaAllocator cuda_allocator;
#endif

#if defined(KL_ENABLE_ROCM)
                static RocmAllocator rocm_allocator;
#endif

                switch (device.type())
                {
                case DeviceType::CPU:
                        return cpu_allocator;

                case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
                        return cuda_allocator;
#else
                        throw std::runtime_error("CUDA backend is not enabled");
#endif

                case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
                        return rocm_allocator;
#else
                        throw std::runtime_error("ROCm backend is not enabled");
#endif

                default:
                        throw std::runtime_error("unknown DeviceType");
                }
        }

}