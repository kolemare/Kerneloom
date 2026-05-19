#include <kernels/rocm/activation/backward_sigmoid_rocm_float32.hiph>

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

        __global__ void backward_sigmoid_rocm_float32_kernel(
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

            const float y = activation_output[index];

            grad[index] *= y * (1.0f - y);
        }

    }

    void backward_sigmoid_rocm_float32(
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
            backward_sigmoid_rocm_float32_kernel,
            grid,
            block,
            0,
            0,
            activation_output_data,
            grad_data,
            count);

        check_hip(hipGetLastError(), "ROCm Sigmoid backward kernel launch failed");
        check_hip(hipDeviceSynchronize(), "ROCm Sigmoid backward synchronization failed");
    }

}