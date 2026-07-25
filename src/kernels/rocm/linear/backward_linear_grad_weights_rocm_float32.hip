#include <kernels/rocm/linear/backward_linear_grad_weights_rocm_float32.hiph>

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

        constexpr int output_tile_size =
            32;

        constexpr int input_tile_size =
            32;

        constexpr int batch_tile_size =
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

        __global__ void backward_linear_grad_weights_rocm_float32_kernel(
            const float *__restrict__ input,
            const float *__restrict__ grad_output,
            float *__restrict__ grad_weights,
            const std::uint32_t batch_size,
            const std::uint32_t input_features,
            const std::uint32_t output_features)
        {
            __shared__ float grad_output_tile
                [batch_tile_size]
                [output_tile_size];

            __shared__ float input_tile
                [batch_tile_size]
                [input_tile_size];

            const int thread_x =
                static_cast<int>(threadIdx.x);

            const int thread_y =
                static_cast<int>(threadIdx.y);

            /*
             * Grad-weight values calculated by this thread.
             */

            const std::uint32_t output_0 =
                static_cast<std::uint32_t>(blockIdx.y) *
                    output_tile_size +
                static_cast<std::uint32_t>(thread_y);

            const std::uint32_t output_1 =
                output_0 + block_size;

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

            for (std::uint32_t batch_base = 0;
                 batch_base < batch_size;
                 batch_base += batch_tile_size)
            {
                const std::uint32_t batch =
                    batch_base +
                    static_cast<std::uint32_t>(thread_y);

                const std::uint32_t output_load_0 =
                    static_cast<std::uint32_t>(blockIdx.y) *
                        output_tile_size +
                    static_cast<std::uint32_t>(thread_x);

                const std::uint32_t output_load_1 =
                    output_load_0 + block_size;

                const std::uint32_t input_load_0 =
                    static_cast<std::uint32_t>(blockIdx.x) *
                        input_tile_size +
                    static_cast<std::uint32_t>(thread_x);

                const std::uint32_t input_load_1 =
                    input_load_0 + block_size;

                if (batch < batch_size &&
                    output_load_0 < output_features)
                {
                    grad_output_tile[thread_y][thread_x] =
                        grad_output[batch * output_features +
                                    output_load_0];
                }
                else
                {
                    grad_output_tile[thread_y][thread_x] =
                        0.0F;
                }

                if (batch < batch_size &&
                    output_load_1 < output_features)
                {
                    grad_output_tile
                        [thread_y]
                        [thread_x + block_size] =
                            grad_output[batch *
                                            output_features +
                                        output_load_1];
                }
                else
                {
                    grad_output_tile
                        [thread_y]
                        [thread_x + block_size] =
                            0.0F;
                }

                if (batch < batch_size &&
                    input_load_0 < input_features)
                {
                    input_tile[thread_y][thread_x] =
                        input[batch * input_features +
                              input_load_0];
                }
                else
                {
                    input_tile[thread_y][thread_x] =
                        0.0F;
                }

                if (batch < batch_size &&
                    input_load_1 < input_features)
                {
                    input_tile
                        [thread_y]
                        [thread_x + block_size] =
                            input[batch *
                                      input_features +
                                  input_load_1];
                }
                else
                {
                    input_tile
                        [thread_y]
                        [thread_x + block_size] =
                            0.0F;
                }

                __syncthreads();

                for (int k = 0;
                     k < batch_tile_size;
                     ++k)
                {
                    const float grad_output_0 =
                        grad_output_tile[k][thread_y];

                    const float grad_output_1 =
                        grad_output_tile
                            [k]
                            [thread_y + block_size];

                    const float input_value_0 =
                        input_tile[k][thread_x];

                    const float input_value_1 =
                        input_tile
                            [k]
                            [thread_x + block_size];

                    sum_00 = fmaf(
                        grad_output_0,
                        input_value_0,
                        sum_00);

                    sum_01 = fmaf(
                        grad_output_0,
                        input_value_1,
                        sum_01);

                    sum_10 = fmaf(
                        grad_output_1,
                        input_value_0,
                        sum_10);

                    sum_11 = fmaf(
                        grad_output_1,
                        input_value_1,
                        sum_11);
                }

                __syncthreads();
            }

            if (output_0 < output_features &&
                input_0 < input_features)
            {
                grad_weights[output_0 * input_features +
                             input_0] =
                    sum_00;
            }

            if (output_0 < output_features &&
                input_1 < input_features)
            {
                grad_weights[output_0 * input_features +
                             input_1] =
                    sum_01;
            }

            if (output_1 < output_features &&
                input_0 < input_features)
            {
                grad_weights[output_1 * input_features +
                             input_0] =
                    sum_10;
            }

            if (output_1 < output_features &&
                input_1 < input_features)
            {
                grad_weights[output_1 * input_features +
                             input_1] =
                    sum_11;
            }
        }

    }

    void backward_linear_grad_weights_rocm_float32(
        const Tensor &input,
        const Tensor &grad_output,
        Tensor &grad_weights)
    {
        const auto batch_size =
            static_cast<std::uint32_t>(input.shape()[0]);

        const auto input_features =
            static_cast<std::uint32_t>(input.shape()[1]);

        const auto output_features =
            static_cast<std::uint32_t>(grad_output.shape()[1]);

        if (batch_size == 0U ||
            input_features == 0U ||
            output_features == 0U)
        {
            return;
        }

        const auto *input_data =
            static_cast<const float *>(input.data());

        const auto *grad_output_data =
            static_cast<const float *>(grad_output.data());

        auto *grad_weights_data =
            static_cast<float *>(grad_weights.data());

        const dim3 block(
            block_size,
            block_size);

        const dim3 grid(
            divide_up(
                input_features,
                input_tile_size),
            divide_up(
                output_features,
                output_tile_size));

        hipLaunchKernelGGL(
            backward_linear_grad_weights_rocm_float32_kernel,
            grid,
            block,
            0,
            0,
            input_data,
            grad_output_data,
            grad_weights_data,
            batch_size,
            input_features,
            output_features);

        check_hip(
            hipGetLastError(),
            "ROCm backward linear grad_weights kernel launch failed");
    }

}