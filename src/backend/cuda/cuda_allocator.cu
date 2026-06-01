#include <backend/cuda/cuda_allocator.cuh>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace kl
{

    namespace
    {

        void check_cuda(cudaError_t error, const char *message)
        {
            if (error != cudaSuccess)
            {
                throw std::runtime_error(
                    std::string(message) + ": " + cudaGetErrorString(error));
            }
        }

    }

    void *CudaAllocator::allocate(std::size_t nbytes)
    {
        if (nbytes == 0)
        {
            return nullptr;
        }

        void *ptr = nullptr;
        check_cuda(cudaMalloc(&ptr, nbytes), "cudaMalloc failed");

        return ptr;
    }

    void CudaAllocator::deallocate(
        void *ptr) noexcept
    {
        if (ptr != nullptr)
        {
            static_cast<void>(
                cudaFree(ptr));
        }
    }

}