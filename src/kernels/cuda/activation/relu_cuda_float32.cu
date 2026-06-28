#include <kernels/cuda/activation/relu_cuda_float32.cuh>

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

        __global__ void relu_cuda_float32_kernel(float *data, std::size_t count)
        {
            const std::size_t index =
                blockIdx.x * blockDim.x + threadIdx.x;

            if (index >= count)
            {
                return;
            }

            if (data[index] < 0.0f)
            {
                data[index] = 0.0f;
            }
        }

    }

    void relu_cuda_float32(Tensor &tensor)
    {
        float *data = static_cast<float *>(tensor.data());

        const std::size_t count = tensor.numel();

        constexpr int block_size = 256;

        dim3 block(block_size);
        dim3 grid(static_cast<unsigned int>((count + block_size - 1) / block_size));

        relu_cuda_float32_kernel<<<grid, block>>>(data, count);

        check_cuda(cudaGetLastError(), "CUDA ReLU kernel launch failed");
    }

}