#include <kernels/rocm/activation/sigmoid_rocm_float32.hiph>

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

        __global__ void sigmoid_rocm_float32_kernel(float *data, std::size_t count)
        {
            const std::size_t index =
                blockIdx.x * blockDim.x + threadIdx.x;

            if (index >= count)
            {
                return;
            }

            data[index] = 1.0f / (1.0f + expf(-data[index]));
        }

    }

    void sigmoid_rocm_float32(Tensor &tensor)
    {
        float *data = static_cast<float *>(tensor.data());

        const std::size_t count = tensor.numel();

        constexpr int block_size = 256;

        dim3 block(block_size);
        dim3 grid(static_cast<unsigned int>((count + block_size - 1) / block_size));

        hipLaunchKernelGGL(
            sigmoid_rocm_float32_kernel,
            grid,
            block,
            0,
            0,
            data,
            count);

        check_hip(hipGetLastError(), "ROCm Sigmoid kernel launch failed");
        check_hip(hipDeviceSynchronize(), "ROCm Sigmoid synchronization failed");
    }

}