#include <kernels/rocm/losses/backward_binary_cross_entropy_rocm_float32.hiph>

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

        __global__ void backward_binary_cross_entropy_rocm_float32_kernel(
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

            constexpr float epsilon =
                1.0e-7f;

            const float probability =
                fminf(
                    fmaxf(
                        prediction[index],
                        epsilon),
                    1.0f - epsilon);

            grad_prediction[index] =
                scale *
                (-target[index] /
                     probability +
                 (1.0f - target[index]) /
                     (1.0f - probability));
        }

    }

    void backward_binary_cross_entropy_rocm_float32(
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

        float scale =
            1.0f;

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
            backward_binary_cross_entropy_rocm_float32_kernel,
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
            "ROCm backward binary cross entropy kernel launch failed");

        check_hip(
            hipDeviceSynchronize(),
            "ROCm backward binary cross entropy synchronization failed");
    }

}