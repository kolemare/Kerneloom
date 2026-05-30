#include <kernels/rocm/losses/backward_categorical_cross_entropy_rocm_float32.hiph>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
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

        __global__ void backward_categorical_cross_entropy_rocm_float32_kernel(
            const float *prediction,
            const std::int32_t *target,
            float *grad_prediction,
            float scale,
            std::size_t batch_size,
            std::size_t class_count)
        {
            const std::size_t n =
                blockIdx.x * blockDim.x +
                threadIdx.x;

            if (n >= batch_size)
            {
                return;
            }

            const std::int32_t target_class =
                target[n];

            if (target_class < 0 ||
                target_class >=
                    static_cast<std::int32_t>(
                        class_count))
            {
                return;
            }

            constexpr float epsilon =
                1.0e-7f;

            const std::size_t index =
                n * class_count +
                static_cast<std::size_t>(
                    target_class);

            const float probability =
                fminf(
                    fmaxf(
                        prediction[index],
                        epsilon),
                    1.0f - epsilon);

            grad_prediction[index] =
                -scale /
                probability;
        }

    }

    void backward_categorical_cross_entropy_rocm_float32(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &grad_prediction,
        Reduction reduction)
    {
        const std::size_t batch_size =
            prediction.shape()[0];

        const std::size_t class_count =
            prediction.shape()[1];

        const float *prediction_data =
            static_cast<const float *>(
                prediction.data());

        const std::int32_t *target_data =
            static_cast<const std::int32_t *>(
                target.data());

        float *grad_prediction_data =
            static_cast<float *>(
                grad_prediction.data());

        float scale =
            1.0f;

        if (reduction == Reduction::Mean)
        {
            scale /=
                static_cast<float>(
                    batch_size);
        }

        check_hip(
            hipMemset(
                grad_prediction_data,
                0,
                grad_prediction.nbytes()),
            "ROCm categorical cross entropy gradient initialization failed");

        constexpr unsigned int block_size =
            256;

        const dim3 block(
            block_size);

        const dim3 grid(
            static_cast<unsigned int>(
                (batch_size +
                 block_size -
                 1) /
                block_size));

        hipLaunchKernelGGL(
            backward_categorical_cross_entropy_rocm_float32_kernel,
            grid,
            block,
            0,
            0,
            prediction_data,
            target_data,
            grad_prediction_data,
            scale,
            batch_size,
            class_count);

        check_hip(
            hipGetLastError(),
            "ROCm backward categorical cross entropy kernel launch failed");

        check_hip(
            hipDeviceSynchronize(),
            "ROCm backward categorical cross entropy synchronization failed");
    }

}