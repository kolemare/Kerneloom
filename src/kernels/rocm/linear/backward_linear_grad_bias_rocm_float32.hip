#include <kernels/rocm/linear/backward_linear_grad_bias_rocm_float32.hiph>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace kl
{

    namespace
    {

        constexpr int output_tile_size =
            64;

        constexpr int reduction_lanes =
            8;

        void check_hip(
            const hipError_t error,
            const char *message)
        {
            if (error != hipSuccess)
            {
                throw std::runtime_error(
                    std::string(message) + ": " +
                    hipGetErrorString(error));
            }
        }

        constexpr unsigned int divide_up(
            const std::uint32_t value,
            const std::uint32_t divisor)
        {
            return static_cast<unsigned int>(
                (value + divisor - 1U) / divisor);
        }

        __global__ void backward_linear_grad_bias_rocm_float32_kernel(
            const float *__restrict__ grad_output,
            float *__restrict__ grad_bias,
            const std::uint32_t batch_size,
            const std::uint32_t output_features)
        {
            __shared__ float partial_sums
                [reduction_lanes]
                [output_tile_size];

            const int thread_x =
                static_cast<int>(threadIdx.x);

            const int thread_y =
                static_cast<int>(threadIdx.y);

            const std::uint32_t output =
                static_cast<std::uint32_t>(blockIdx.x) *
                    output_tile_size +
                static_cast<std::uint32_t>(thread_x);

            float sum =
                0.0F;

            if (output < output_features)
            {
                for (std::uint32_t batch =
                         static_cast<std::uint32_t>(thread_y);
                     batch < batch_size;
                     batch += reduction_lanes)
                {
                    sum = fmaf(
                        1.0F,
                        grad_output[batch * output_features +
                                    output],
                        sum);
                }
            }

            partial_sums[thread_y][thread_x] =
                sum;

            __syncthreads();

            if (thread_y == 0 &&
                output < output_features)
            {
                float total =
                    partial_sums[0][thread_x];

                for (int lane = 1;
                     lane < reduction_lanes;
                     ++lane)
                {
                    total += partial_sums[lane][thread_x];
                }

                grad_bias[output] =
                    total;
            }
        }

    }

    void backward_linear_grad_bias_rocm_float32(
        const Tensor &grad_output,
        Tensor &grad_bias)
    {
        const auto batch_size =
            static_cast<std::uint32_t>(grad_output.shape()[0]);

        const auto output_features =
            static_cast<std::uint32_t>(grad_output.shape()[1]);

        if (batch_size == 0U ||
            output_features == 0U)
        {
            return;
        }

        const auto *grad_output_data =
            static_cast<const float *>(grad_output.data());

        auto *grad_bias_data =
            static_cast<float *>(grad_bias.data());

        const dim3 block(
            output_tile_size,
            reduction_lanes);

        const dim3 grid(
            divide_up(
                output_features,
                output_tile_size));

        hipLaunchKernelGGL(
            backward_linear_grad_bias_rocm_float32_kernel,
            grid,
            block,
            0,
            0,
            grad_output_data,
            grad_bias_data,
            batch_size,
            output_features);

        check_hip(
            hipGetLastError(),
            "ROCm backward linear grad_bias kernel launch failed");
    }

}