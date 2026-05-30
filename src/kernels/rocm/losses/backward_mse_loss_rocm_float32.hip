#include <kernels/rocm/losses/backward_mse_loss_rocm_float32.hiph>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace kl
{

    namespace
    {

        void check_hip(
            hipError_t error,
            const char *message)
        {
            if (error != hipSuccess)
            {
                throw std::runtime_error(
                    std::string(message) +
                    ": " +
                    hipGetErrorString(error));
            }
        }

        __global__ void backward_mse_loss_rocm_float32_kernel(
            const float *prediction,
            const float *target,
            float *grad_prediction,
            float scale,
            std::size_t count)
        {
            const std::size_t index =
                blockIdx.x * blockDim.x +
                threadIdx.x;

            if (index >= count)
            {
                return;
            }

            grad_prediction[index] =
                scale *
                (prediction[index] -
                 target[index]);
        }

    }

    void backward_mse_loss_rocm_float32(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &grad_prediction,
        Reduction reduction)
    {
        const float *prediction_data =
            static_cast<const float *>(
                prediction.data());

        const float *target_data =
            static_cast<const float *>(
                target.data());

        float *grad_prediction_data =
            static_cast<float *>(
                grad_prediction.data());

        const std::size_t count =
            prediction.numel();

        float scale = 2.0f;

        if (reduction == Reduction::Mean)
        {
            scale /=
                static_cast<float>(count);
        }

        constexpr unsigned int block_size =
            256;

        const dim3 block(
            block_size);

        const dim3 grid(
            static_cast<unsigned int>(
                (count + block_size - 1) /
                block_size));

        hipLaunchKernelGGL(
            backward_mse_loss_rocm_float32_kernel,
            grid,
            block,
            0,
            0,
            prediction_data,
            target_data,
            grad_prediction_data,
            scale,
            count);

        check_hip(
            hipGetLastError(),
            "ROCm backward MSE kernel launch failed");

        check_hip(
            hipDeviceSynchronize(),
            "ROCm backward MSE synchronization failed");
    }

}