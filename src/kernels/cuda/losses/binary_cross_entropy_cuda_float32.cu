#include <kernels/cuda/losses/binary_cross_entropy_cuda_float32.cuh>

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace kl
{

    namespace
    {

        void check_cuda(
            cudaError_t error,
            const char *message)
        {
            if (error !=
                cudaSuccess)
            {
                throw std::runtime_error(
                    std::string(message) +
                    ": " +
                    cudaGetErrorString(
                        error));
            }
        }

        __global__ void binary_cross_entropy_cuda_float32_kernel(
            const float *prediction,
            const float *target,
            float *result,
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

            constexpr float epsilon =
                1.0e-7f;

            const float probability =
                fminf(
                    fmaxf(
                        prediction[index],
                        epsilon),
                    1.0f -
                        epsilon);

            const float target_value =
                target[index];

            const float loss =
                -target_value *
                    logf(
                        probability) -
                (1.0f -
                 target_value) *
                    logf(
                        1.0f -
                        probability);

            atomicAdd(
                result,
                scale *
                    loss);
        }

    }

    void binary_cross_entropy_cuda_float32(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &result,
        Reduction reduction,
        std::size_t valid_sample_count)
    {
        const std::size_t elements_per_sample =
            prediction.numel() /
            prediction.shape()[0];

        const std::size_t count =
            valid_sample_count *
            elements_per_sample;

        const float *prediction_data =
            static_cast<const float *>(
                prediction.data());

        const float *target_data =
            static_cast<const float *>(
                target.data());

        float *result_data =
            static_cast<float *>(
                result.data());

        float scale =
            1.0f;

        if (reduction ==
            Reduction::Mean)
        {
            scale /=
                static_cast<float>(
                    count);
        }

        check_cuda(
            cudaMemset(
                result_data,
                0,
                sizeof(float)),
            "CUDA binary cross entropy result initialization failed");

        constexpr unsigned int block_size =
            256;

        const dim3 block(
            block_size);

        const dim3 grid(
            static_cast<unsigned int>(
                (count +
                 block_size -
                 1) /
                block_size));

        binary_cross_entropy_cuda_float32_kernel<<<
            grid,
            block>>>(
            prediction_data,
            target_data,
            result_data,
            scale,
            count);

        check_cuda(
            cudaGetLastError(),
            "CUDA binary cross entropy kernel launch failed");
    }

}