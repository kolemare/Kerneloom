#include <core/allocator.hpp>

#include <backend/cpu/cpu_allocator.hpp>

#if defined(KL_ENABLE_CUDA)
#include <backend/cuda/cuda_allocator.cuh>
#include <backend/cuda/cuda_pinned_allocator.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <backend/rocm/rocm_allocator.hiph>
#include <backend/rocm/rocm_pinned_allocator.hiph>
#endif

#include <stdexcept>

namespace kl
{

        Allocator &allocator_for(
            Device device,
            MemoryType memory_type)
        {
                static CpuAllocator
                    cpu_allocator;

#if defined(KL_ENABLE_CUDA)
                static CudaAllocator
                    cuda_allocator;

                static CudaPinnedAllocator
                    cuda_pinned_allocator;
#endif

#if defined(KL_ENABLE_ROCM)
                static RocmAllocator
                    rocm_allocator;

                static RocmPinnedAllocator
                    rocm_pinned_allocator;
#endif

                if (memory_type ==
                    MemoryType::CudaPinnedHost)
                {
                        if (device.type() !=
                            DeviceType::CPU)
                        {
                                throw std::runtime_error(
                                    "CUDA pinned memory must use CPU device");
                        }

#if defined(KL_ENABLE_CUDA)
                        return cuda_pinned_allocator;
#else
                        throw std::runtime_error(
                            "CUDA pinned memory requested but CUDA backend is not enabled");
#endif
                }

                if (memory_type ==
                    MemoryType::RocmPinnedHost)
                {
                        if (device.type() !=
                            DeviceType::CPU)
                        {
                                throw std::runtime_error(
                                    "ROCm pinned memory must use CPU device");
                        }

#if defined(KL_ENABLE_ROCM)
                        return rocm_pinned_allocator;
#else
                        throw std::runtime_error(
                            "ROCm pinned memory requested but ROCm backend is not enabled");
#endif
                }

                switch (device.type())
                {
                case DeviceType::CPU:
                        return cpu_allocator;

                case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
                        return cuda_allocator;
#else
                        throw std::runtime_error(
                            "CUDA backend is not enabled");
#endif

                case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
                        return rocm_allocator;
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