#include <kernels/cuda/losses/categorical_cross_entropy_cuda_float32.cuh>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
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

        __global__ void categorical_cross_entropy_cuda_float32_kernel(
            const float *prediction,
            const std::int32_t *target,
            float *result,
            float scale,
            std::size_t valid_sample_count,
            std::size_t class_count)
        {
            const std::size_t n =
                blockIdx.x *
                    blockDim.x +
                threadIdx.x;

            if (n >=
                valid_sample_count)
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
                n *
                    class_count +
                static_cast<std::size_t>(
                    target_class);

            const float probability =
                fminf(
                    fmaxf(
                        prediction[index],
                        epsilon),
                    1.0f -
                        epsilon);

            atomicAdd(
                result,
                -scale *
                    logf(
                        probability));
        }

    }

    void categorical_cross_entropy_cuda_float32(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &result,
        Reduction reduction,
        std::size_t valid_sample_count)
    {
        const std::size_t class_count =
            prediction.shape()[1];

        const float *prediction_data =
            static_cast<const float *>(
                prediction.data());

        const std::int32_t *target_data =
            static_cast<const std::int32_t *>(
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
                    valid_sample_count);
        }

        check_cuda(
            cudaMemset(
                result_data,
                0,
                sizeof(float)),
            "CUDA categorical cross entropy result initialization failed");

        constexpr unsigned int block_size =
            256;

        const dim3 block(
            block_size);

        const dim3 grid(
            static_cast<unsigned int>(
                (valid_sample_count +
                 block_size -
                 1) /
                block_size));

        categorical_cross_entropy_cuda_float32_kernel<<<
            grid,
            block>>>(
            prediction_data,
            target_data,
            result_data,
            scale,
            valid_sample_count,
            class_count);

        check_cuda(
            cudaGetLastError(),
            "CUDA categorical cross entropy kernel launch failed");
    }

}