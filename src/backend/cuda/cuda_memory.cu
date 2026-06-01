#include <backend/cuda/cuda_memory.cuh>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace kl
{

    std::size_t available_cuda_memory_bytes()
    {
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;

        const cudaError_t error =
            cudaMemGetInfo(
                &free_bytes,
                &total_bytes);

        if (error != cudaSuccess)
        {
            throw std::runtime_error(
                std::string(
                    "cudaMemGetInfo failed: ") +
                cudaGetErrorString(
                    error));
        }

        return free_bytes;
    }

}