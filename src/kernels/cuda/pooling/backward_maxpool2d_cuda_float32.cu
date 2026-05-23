#include <kernels/cuda/pooling/backward_maxpool2d_cuda_float32.cuh>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
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

        __global__ void zero_float32_kernel(
            float *data,
            std::size_t count)
        {
            const std::size_t index =
                blockIdx.x * blockDim.x + threadIdx.x;

            if (index < count)
            {
                data[index] = 0.0f;
            }
        }

        __global__ void backward_maxpool2d_cuda_float32_kernel(
            const std::int32_t *indices,
            const float *grad_output,
            float *grad_input,
            std::size_t count)
        {
            const std::size_t index =
                blockIdx.x * blockDim.x + threadIdx.x;

            if (index >= count)
            {
                return;
            }

            const std::int32_t input_index = indices[index];

            atomicAdd(
                &grad_input[input_index],
                grad_output[index]);
        }

    }

    void backward_maxpool2d_cuda_float32(
        const Tensor &indices,
        const Tensor &grad_output,
        Tensor &grad_input)
    {
        const std::int32_t *indices_data =
            static_cast<const std::int32_t *>(indices.data());

        const float *grad_output_data =
            static_cast<const float *>(grad_output.data());

        float *grad_input_data =
            static_cast<float *>(grad_input.data());

        constexpr int block_size = 256;

        const std::size_t grad_input_count = grad_input.numel();
        const std::size_t grad_output_count = grad_output.numel();

        dim3 zero_block(block_size);
        dim3 zero_grid(
            static_cast<unsigned int>(
                (grad_input_count + block_size - 1) / block_size));

        zero_float32_kernel<<<zero_grid, zero_block>>>(
            grad_input_data,
            grad_input_count);

        check_cuda(
            cudaGetLastError(),
            "CUDA backward maxpool2d zero kernel launch failed");

        dim3 scatter_block(block_size);
        dim3 scatter_grid(
            static_cast<unsigned int>(
                (grad_output_count + block_size - 1) / block_size));

        backward_maxpool2d_cuda_float32_kernel<<<scatter_grid, scatter_block>>>(
            indices_data,
            grad_output_data,
            grad_input_data,
            grad_output_count);

        check_cuda(
            cudaGetLastError(),
            "CUDA backward maxpool2d kernel launch failed");

        check_cuda(
            cudaDeviceSynchronize(),
            "CUDA backward maxpool2d synchronization failed");
    }

}