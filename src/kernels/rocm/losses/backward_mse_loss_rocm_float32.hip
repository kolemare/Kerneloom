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
            if (error !=
                hipSuccess)
            {
                throw std::runtime_error(
                    std::string(message) +
                    ": " +
                    hipGetErrorString(
                        error));
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
                blockIdx.x *
                    blockDim.x +
                threadIdx.x;

            if (index >=
                count)
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
        Reduction reduction,
        std::size_t valid_sample_count)
    {
        const std::size_t elements_per_sample =
            prediction.numel() /
            prediction.shape()[0];

        const std::size_t valid_element_count =
            valid_sample_count *
            elements_per_sample;

        const float *prediction_data =
            static_cast<const float *>(
                prediction.data());

        const float *target_data =
            static_cast<const float *>(
                target.data());

        float *grad_prediction_data =
            static_cast<float *>(
                grad_prediction.data());

        float scale =
            2.0f;

        if (reduction ==
            Reduction::Mean)
        {
            scale /=
                static_cast<float>(
                    valid_element_count);
        }

        if (valid_element_count <
            grad_prediction.numel())
        {
            check_hip(
                hipMemset(
                    grad_prediction_data +
                        valid_element_count,
                    0,
                    (grad_prediction.numel() -
                     valid_element_count) *
                        sizeof(float)),
                "ROCm padded MSE gradient initialization failed");
        }

        constexpr unsigned int block_size =
            256;

        const dim3 block(
            block_size);

        const dim3 grid(
            static_cast<unsigned int>(
                (valid_element_count +
                 block_size -
                 1) /
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
            valid_element_count);

        check_hip(
            hipGetLastError(),
            "ROCm backward MSE kernel launch failed");
    }

}