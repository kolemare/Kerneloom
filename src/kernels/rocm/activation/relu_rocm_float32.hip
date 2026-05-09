#include <kernels/rocm/activation/relu_rocm_float32.hiph>

#include <hip/hip_runtime.h>

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

        __global__ void relu_rocm_float32_kernel(float *data, std::size_t count)
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

    void relu_rocm_float32(Tensor &tensor)
    {
        float *data = static_cast<float *>(tensor.data());

        const std::size_t count = tensor.numel();

        constexpr int block_size = 256;

        dim3 block(block_size);
        dim3 grid(static_cast<unsigned int>((count + block_size - 1) / block_size));

        hipLaunchKernelGGL(
            relu_rocm_float32_kernel,
            grid,
            block,
            0,
            0,
            data,
            count);

        check_hip(hipGetLastError(), "ROCm ReLU kernel launch failed");
        check_hip(hipDeviceSynchronize(), "ROCm ReLU synchronization failed");
    }

}