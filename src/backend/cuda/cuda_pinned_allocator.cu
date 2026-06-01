#include <backend/cuda/cuda_pinned_allocator.cuh>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace kl
{

    namespace
    {

        void check_cuda(
            cudaError_t error,
            const char *message)
        {
            if (error != cudaSuccess)
            {
                throw std::runtime_error(
                    std::string(message) +
                    ": " +
                    cudaGetErrorString(error));
            }
        }

    }

    void *CudaPinnedAllocator::allocate(
        std::size_t nbytes)
    {
        if (nbytes == 0)
        {
            return nullptr;
        }

        void *ptr =
            nullptr;

        check_cuda(
            cudaHostAlloc(
                &ptr,
                nbytes,
                cudaHostAllocDefault),
            "cudaHostAlloc failed");

        return ptr;
    }

    void CudaPinnedAllocator::deallocate(
        void *ptr)
    {
        if (ptr == nullptr)
        {
            return;
        }

        check_cuda(
            cudaFreeHost(
                ptr),
            "cudaFreeHost failed");
    }

}