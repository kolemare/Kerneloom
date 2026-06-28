#include <kernels/rocm/pooling/backward_maxpool2d_rocm_float32.hiph>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace kl
{

    namespace
    {

        void check_hip(hipError_t error, const char *message)
        {
            if (error != hipSuccess)
            {
                throw std::runtime_error(
                    std::string(message) + ": " + hipGetErrorString(error));
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

        __global__ void backward_maxpool2d_rocm_float32_kernel(
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

    void backward_maxpool2d_rocm_float32(
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

        hipLaunchKernelGGL(
            zero_float32_kernel,
            zero_grid,
            zero_block,
            0,
            0,
            grad_input_data,
            grad_input_count);

        check_hip(
            hipGetLastError(),
            "ROCm backward maxpool2d zero kernel launch failed");

        dim3 scatter_block(block_size);
        dim3 scatter_grid(
            static_cast<unsigned int>(
                (grad_output_count + block_size - 1) / block_size));

        hipLaunchKernelGGL(
            backward_maxpool2d_rocm_float32_kernel,
            scatter_grid,
            scatter_block,
            0,
            0,
            indices_data,
            grad_output_data,
            grad_input_data,
            grad_output_count);

        check_hip(
            hipGetLastError(),
            "ROCm backward maxpool2d kernel launch failed");
    }

}