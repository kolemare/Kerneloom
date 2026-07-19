#include <kernels/rocm/linear/linear_rocm_float32.hiph>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace kl
{

    namespace
    {

        constexpr int block_size = 16;
        constexpr int output_tile_size = 32;
        constexpr int feature_tile_size = 16;

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

        template <bool has_bias>
        __global__ void linear_rocm_float32_kernel(
            const float *__restrict__ input,
            const float *__restrict__ weights,
            const float *__restrict__ bias,
            float *__restrict__ result,
            const std::uint32_t batch_size,
            const std::uint32_t input_features,
            const std::uint32_t output_features)
        {
            __shared__ float input_tile
                [output_tile_size]
                [feature_tile_size + 1];

            __shared__ float weight_tile
                [output_tile_size]
                [feature_tile_size + 1];

            const int thread_x =
                static_cast<int>(threadIdx.x);

            const int thread_y =
                static_cast<int>(threadIdx.y);

            const std::uint32_t batch_base =
                static_cast<std::uint32_t>(blockIdx.y) *
                output_tile_size;

            const std::uint32_t output_base =
                static_cast<std::uint32_t>(blockIdx.x) *
                output_tile_size;

            const std::uint32_t batch_0 =
                batch_base +
                static_cast<std::uint32_t>(thread_y);

            const std::uint32_t batch_1 =
                batch_0 + block_size;

            const std::uint32_t output_0 =
                output_base +
                static_cast<std::uint32_t>(thread_x);

            const std::uint32_t output_1 =
                output_0 + block_size;

            const std::uint32_t weight_output_0 =
                output_base +
                static_cast<std::uint32_t>(thread_y);

            const std::uint32_t weight_output_1 =
                weight_output_0 + block_size;

            float sum_00 = 0.0F;
            float sum_01 = 0.0F;
            float sum_10 = 0.0F;
            float sum_11 = 0.0F;

            for (std::uint32_t feature_base = 0;
                 feature_base < input_features;
                 feature_base += feature_tile_size)
            {
                const std::uint32_t feature =
                    feature_base +
                    static_cast<std::uint32_t>(thread_x);

                if (batch_0 < batch_size &&
                    feature < input_features)
                {
                    input_tile[thread_y][thread_x] =
                        input[batch_0 * input_features +
                              feature];
                }
                else
                {
                    input_tile[thread_y][thread_x] = 0.0F;
                }

                if (batch_1 < batch_size &&
                    feature < input_features)
                {
                    input_tile
                        [thread_y + block_size]
                        [thread_x] =
                            input[batch_1 * input_features +
                                  feature];
                }
                else
                {
                    input_tile
                        [thread_y + block_size]
                        [thread_x] = 0.0F;
                }

                if (weight_output_0 < output_features &&
                    feature < input_features)
                {
                    weight_tile[thread_y][thread_x] =
                        weights[weight_output_0 *
                                    input_features +
                                feature];
                }
                else
                {
                    weight_tile[thread_y][thread_x] = 0.0F;
                }

                if (weight_output_1 < output_features &&
                    feature < input_features)
                {
                    weight_tile
                        [thread_y + block_size]
                        [thread_x] =
                            weights[weight_output_1 *
                                        input_features +
                                    feature];
                }
                else
                {
                    weight_tile
                        [thread_y + block_size]
                        [thread_x] = 0.0F;
                }

                __syncthreads();

                for (int k = 0;
                     k < feature_tile_size;
                     ++k)
                {
                    const float input_0 =
                        input_tile[thread_y][k];

                    const float input_1 =
                        input_tile
                            [thread_y + block_size]
                            [k];

                    const float weight_0 =
                        weight_tile[thread_x][k];

                    const float weight_1 =
                        weight_tile
                            [thread_x + block_size]
                            [k];

                    sum_00 = fmaf(
                        input_0,
                        weight_0,
                        sum_00);

                    sum_01 = fmaf(
                        input_0,
                        weight_1,
                        sum_01);

                    sum_10 = fmaf(
                        input_1,
                        weight_0,
                        sum_10);

                    sum_11 = fmaf(
                        input_1,
                        weight_1,
                        sum_11);
                }

                __syncthreads();
            }

            if constexpr (has_bias)
            {
                if (output_0 < output_features)
                {
                    sum_00 += bias[output_0];
                    sum_10 += bias[output_0];
                }

                if (output_1 < output_features)
                {
                    sum_01 += bias[output_1];
                    sum_11 += bias[output_1];
                }
            }

            if (batch_0 < batch_size &&
                output_0 < output_features)
            {
                result[batch_0 * output_features +
                       output_0] = sum_00;
            }

            if (batch_0 < batch_size &&
                output_1 < output_features)
            {
                result[batch_0 * output_features +
                       output_1] = sum_01;
            }

            if (batch_1 < batch_size &&
                output_0 < output_features)
            {
                result[batch_1 * output_features +
                       output_0] = sum_10;
            }

            if (batch_1 < batch_size &&
                output_1 < output_features)
            {
                result[batch_1 * output_features +
                       output_1] = sum_11;
            }
        }

    }

    void linear_rocm_float32(
        const Tensor &input,
        const Tensor &weights,
        const Tensor *bias,
        Tensor &result)
    {
        const auto batch_size =
            static_cast<std::uint32_t>(input.shape()[0]);

        const auto input_features =
            static_cast<std::uint32_t>(input.shape()[1]);

        const auto output_features =
            static_cast<std::uint32_t>(weights.shape()[0]);

        if (batch_size == 0U ||
            input_features == 0U ||
            output_features == 0U)
        {
            return;
        }

        const auto *input_data =
            static_cast<const float *>(input.data());

        const auto *weights_data =
            static_cast<const float *>(weights.data());

        const auto *bias_data =
            bias != nullptr
                ? static_cast<const float *>(bias->data())
                : nullptr;

        auto *result_data =
            static_cast<float *>(result.data());

        const dim3 block(
            block_size,
            block_size);

        const dim3 grid(
            divide_up(
                output_features,
                output_tile_size),
            divide_up(
                batch_size,
                output_tile_size));

        if (bias_data != nullptr)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    linear_rocm_float32_kernel<true>),
                grid,
                block,
                0,
                0,
                input_data,
                weights_data,
                bias_data,
                result_data,
                batch_size,
                input_features,
                output_features);
        }
        else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    linear_rocm_float32_kernel<false>),
                grid,
                block,
                0,
                0,
                input_data,
                weights_data,
                nullptr,
                result_data,
                batch_size,
                input_features,
                output_features);
        }

        check_hip(
            hipGetLastError(),
            "ROCm linear float32 kernel launch failed");
    }

}