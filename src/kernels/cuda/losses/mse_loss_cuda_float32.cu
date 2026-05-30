#include <kernels/cuda/losses/mse_loss_cuda_float32.cuh>

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
            if (error != cudaSuccess)
            {
                throw std::runtime_error(
                    std::string(message) +
                    ": " +
                    cudaGetErrorString(error));
            }
        }

        __global__ void mse_loss_cuda_float32_kernel(
            const float *prediction,
            const float *target,
            float *result,
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

            const float difference =
                prediction[index] -
                target[index];

            atomicAdd(
                result,
                scale *
                    difference *
                    difference);
        }

    }

    void mse_loss_cuda_float32(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &result,
        Reduction reduction)
    {
        const float *prediction_data =
            static_cast<const float *>(
                prediction.data());

        const float *target_data =
            static_cast<const float *>(
                target.data());

        float *result_data =
            static_cast<float *>(
                result.data());

        const std::size_t count =
            prediction.numel();

        float scale = 1.0f;

        if (reduction == Reduction::Mean)
        {
            scale /=
                static_cast<float>(count);
        }

        check_cuda(
            cudaMemset(
                result_data,
                0,
                sizeof(float)),
            "CUDA MSE result initialization failed");

        constexpr unsigned int block_size =
            256;

        const dim3 block(
            block_size);

        const dim3 grid(
            static_cast<unsigned int>(
                (count + block_size - 1) /
                block_size));

        mse_loss_cuda_float32_kernel<<<
            grid,
            block>>>(
            prediction_data,
            target_data,
            result_data,
            scale,
            count);

        check_cuda(
            cudaGetLastError(),
            "CUDA MSE kernel launch failed");

        check_cuda(
            cudaDeviceSynchronize(),
            "CUDA MSE synchronization failed");
    }

}