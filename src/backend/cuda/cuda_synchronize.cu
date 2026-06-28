#include <backend/cuda/cuda_synchronize.cuh>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace kl
{

    void cuda_synchronize()
    {
        const cudaError_t error =
            cudaDeviceSynchronize();

        if (error != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("CUDA device synchronization failed: ") +
                cudaGetErrorString(error));
        }
    }

}