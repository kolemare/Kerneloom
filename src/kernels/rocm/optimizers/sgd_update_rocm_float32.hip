#include <kernels/rocm/optimizers/sgd_update_rocm_float32.hiph>

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

        __global__ void sgd_update_rocm_float32_kernel(
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

    void sgd_update_rocm_float32(
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

        hipLaunchKernelGGL(
            sgd_update_rocm_float32_kernel,
            grid,
            block,
            0,
            0,
            value_data,
            grad_data,
            learning_rate,
            count);

        check_hip(
            hipGetLastError(),
            "ROCm SGD update kernel launch failed");

        check_hip(
            hipDeviceSynchronize(),
            "ROCm SGD update synchronization failed");
    }

}