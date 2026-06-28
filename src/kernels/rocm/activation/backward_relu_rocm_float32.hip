#include <kernels/rocm/activation/backward_relu_rocm_float32.hiph>

#include <hip/hip_runtime.h>

#include <cstddef>
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

        __global__ void backward_relu_rocm_float32_kernel(
            const float *activation_output,
            float *grad,
            std::size_t count)
        {
            const std::size_t index =
                blockIdx.x * blockDim.x + threadIdx.x;

            if (index >= count)
            {
                return;
            }

            if (activation_output[index] <= 0.0f)
            {
                grad[index] = 0.0f;
            }
        }

    }

    void backward_relu_rocm_float32(
        const Tensor &activation_output,
        Tensor &grad)
    {
        const float *activation_output_data =
            static_cast<const float *>(activation_output.data());

        float *grad_data =
            static_cast<float *>(grad.data());

        const std::size_t count = grad.numel();

        constexpr int block_size = 256;

        dim3 block(block_size);
        dim3 grid(static_cast<unsigned int>((count + block_size - 1) / block_size));

        hipLaunchKernelGGL(
            backward_relu_rocm_float32_kernel,
            grid,
            block,
            0,
            0,
            activation_output_data,
            grad_data,
            count);

        check_hip(hipGetLastError(), "ROCm ReLU backward kernel launch failed");
    }

}