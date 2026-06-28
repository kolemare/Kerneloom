#include <kernels/cuda/optimizers/sgd_update_cuda_float32.cuh>

#include <cuda_runtime.h>

#include <cstddef>
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

        __global__ void sgd_update_cuda_float32_kernel(
            float *value,
            const float *grad,
            float learning_rate,
            std::size_t count)
        {
            const std::size_t index =
                blockIdx.x * blockDim.x + threadIdx.x;

            if (index >= count)
            {
                return;
            }

            value[index] -= learning_rate * grad[index];
        }

    }

    void sgd_update_cuda_float32(
        Tensor &value,
        const Tensor &grad,
        float learning_rate)
    {
        float *value_data =
            static_cast<float *>(value.data());

        const float *grad_data =
            static_cast<const float *>(grad.data());

        const std::size_t count = value.numel();

        constexpr int block_size = 256;

        dim3 block(block_size);
        dim3 grid(static_cast<unsigned int>(
            (count + block_size - 1) / block_size));

        sgd_update_cuda_float32_kernel<<<grid, block>>>(
            value_data,
            grad_data,
            learning_rate,
            count);

        check_cuda(
            cudaGetLastError(),
            "CUDA SGD update kernel launch failed");
    }

}