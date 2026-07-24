#include <kernels/rocm/linear/backward_linear_grad_input_rocm_float32.hiph>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace kl
{

    namespace
    {

        constexpr int block_size =
            16;

        constexpr int input_tile_size =
            32;

        constexpr int output_tile_size =
            16;

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

        __global__ void backward_linear_grad_input_rocm_float32_kernel(
            const float *__restrict__ weights,
            const float *__restrict__ grad_output,
            float *__restrict__ grad_input,
            const std::uint32_t batch_size,
            const std::uint32_t input_features,
            const std::uint32_t output_features)
        {
            __shared__ float grad_output_tile
                [input_tile_size]
                [output_tile_size];

            __shared__ float weight_tile
                [input_tile_size]
                [output_tile_size];

            const int thread_x =
                static_cast<int>(threadIdx.x);

            const int thread_y =
                static_cast<int>(threadIdx.y);

            /*
             * Grad-input values calculated by this thread.
             */

            const std::uint32_t batch_0 =
                static_cast<std::uint32_t>(blockIdx.y) *
                    input_tile_size +
                static_cast<std::uint32_t>(thread_y);

            const std::uint32_t batch_1 =
                batch_0 + block_size;

            const std::uint32_t input_0 =
                static_cast<std::uint32_t>(blockIdx.x) *
                    input_tile_size +
                static_cast<std::uint32_t>(thread_x);

            const std::uint32_t input_1 =
                input_0 + block_size;

            float sum_00 =
                0.0F;

            float sum_01 =
                0.0F;

            float sum_10 =
                0.0F;

            float sum_11 =
                0.0F;

            for (std::uint32_t output_base = 0;
                 output_base < output_features;
                 output_base += output_tile_size)
            {
                const std::uint32_t output_for_grad =
                    output_base +
                    static_cast<std::uint32_t>(thread_x);

                const std::uint32_t output_for_weight =
                    output_base +
                    static_cast<std::uint32_t>(thread_y);

                if (batch_0 < batch_size &&
                    output_for_grad < output_features)
                {
                    grad_output_tile[thread_y][thread_x] =
                        grad_output[batch_0 * output_features +
                                    output_for_grad];
                }
                else
                {
                    grad_output_tile[thread_y][thread_x] =
                        0.0F;
                }

                if (batch_1 < batch_size &&
                    output_for_grad < output_features)
                {
                    grad_output_tile
                        [thread_y + block_size]
                        [thread_x] =
                            grad_output[batch_1 *
                                            output_features +
                                        output_for_grad];
                }
                else
                {
                    grad_output_tile
                        [thread_y + block_size]
                        [thread_x] =
                            0.0F;
                }

                if (output_for_weight < output_features &&
                    input_0 < input_features)
                {
                    weight_tile[thread_x][thread_y] =
                        weights[output_for_weight *
                                    input_features +
                                input_0];
                }
                else
                {
                    weight_tile[thread_x][thread_y] =
                        0.0F;
                }

                if (output_for_weight < output_features &&
                    input_1 < input_features)
                {
                    weight_tile
                        [thread_x + block_size]
                        [thread_y] =
                            weights[output_for_weight *
                                        input_features +
                                    input_1];
                }
                else
                {
                    weight_tile
                        [thread_x + block_size]
                        [thread_y] =
                            0.0F;
                }

                __syncthreads();

                for (int k = 0;
                     k < output_tile_size;
                     ++k)
                {
                    const float grad_output_0 =
                        grad_output_tile[thread_y][k];

                    const float grad_output_1 =
                        grad_output_tile
                            [thread_y + block_size]
                            [k];

                    const float weight_0 =
                        weight_tile[thread_x][k];

                    const float weight_1 =
                        weight_tile
                            [thread_x + block_size]
                            [k];

                    sum_00 = fmaf(
                        grad_output_0,
                        weight_0,
                        sum_00);

                    sum_01 = fmaf(
                        grad_output_0,
                        weight_1,
                        sum_01);

                    sum_10 = fmaf(
                        grad_output_1,
                        weight_0,
                        sum_10);

                    sum_11 = fmaf(
                        grad_output_1,
                        weight_1,
                        sum_11);
                }

                __syncthreads();
            }

            if (batch_0 < batch_size &&
                input_0 < input_features)
            {
                grad_input[batch_0 * input_features +
                           input_0] =
                    sum_00;
            }

            if (batch_0 < batch_size &&
                input_1 < input_features)
            {
                grad_input[batch_0 * input_features +
                           input_1] =
                    sum_01;
            }

            if (batch_1 < batch_size &&
                input_0 < input_features)
            {
                grad_input[batch_1 * input_features +
                           input_0] =
                    sum_10;
            }

            if (batch_1 < batch_size &&
                input_1 < input_features)
            {
                grad_input[batch_1 * input_features +
                           input_1] =
                    sum_11;
            }
        }

    }

    void backward_linear_grad_input_rocm_float32(
        const Tensor &weights,
        const Tensor &grad_output,
        Tensor &grad_input)
    {
        const auto batch_size =
            static_cast<std::uint32_t>(grad_output.shape()[0]);

        const auto output_features =
            static_cast<std::uint32_t>(grad_output.shape()[1]);

        const auto input_features =
            static_cast<std::uint32_t>(weights.shape()[1]);

        if (batch_size == 0U ||
            input_features == 0U ||
            output_features == 0U)
        {
            return;
        }

        const auto *weights_data =
            static_cast<const float *>(weights.data());

        const auto *grad_output_data =
            static_cast<const float *>(grad_output.data());

        auto *grad_input_data =
            static_cast<float *>(grad_input.data());

        const dim3 block(
            block_size,
            block_size);

        const dim3 grid(
            divide_up(
                input_features,
                input_tile_size),
            divide_up(
                batch_size,
                input_tile_size));

        hipLaunchKernelGGL(
            backward_linear_grad_input_rocm_float32_kernel,
            grid,
            block,
            0,
            0,
            weights_data,
            grad_output_data,
            grad_input_data,
            batch_size,
            input_features,
            output_features);

        check_hip(
            hipGetLastError(),
            "ROCm backward linear grad_input kernel launch failed");
    }

}