#include <kernels/cuda/optimizers/adam_update_cuda_float32.cuh>

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace kl
{

    namespace
    {

        void check_cuda(cudaError_t error, const char *message)
        {
            if (error != cudaSuccess)
            {
                throw std::runtime_error(
                    std::string(message) + ": " + cudaGetErrorString(error));
            }
        }

        __global__ void adam_update_cuda_float32_kernel(
            float *value,
            const float *grad,
            float *first_moment,
            float *second_moment,
            float learning_rate,
            float beta1,
            float beta2,
            float epsilon,
            float beta1_power,
            float beta2_power,
            std::size_t count)
        {
            const std::size_t index =
                blockIdx.x * blockDim.x + threadIdx.x;

            if (index >= count)
            {
                return;
            }

            const float beta1_correction = 1.0f - beta1_power;
            const float beta2_correction = 1.0f - beta2_power;

            const float g = grad[index];

            first_moment[index] =
                beta1 * first_moment[index] +
                (1.0f - beta1) * g;

            second_moment[index] =
                beta2 * second_moment[index] +
                (1.0f - beta2) * g * g;

            const float corrected_first =
                first_moment[index] / beta1_correction;

            const float corrected_second =
                second_moment[index] / beta2_correction;

            value[index] -=
                learning_rate *
                corrected_first /
                (sqrtf(corrected_second) + epsilon);
        }

    }

    void adam_update_cuda_float32(
        Tensor &value,
        const Tensor &grad,
        Tensor &first_moment,
        Tensor &second_moment,
        float learning_rate,
        float beta1,
        float beta2,
        float epsilon,
        float beta1_power,
        float beta2_power)
    {
        float *value_data =
            static_cast<float *>(value.data());

        const float *grad_data =
            static_cast<const float *>(grad.data());

        float *first_moment_data =
            static_cast<float *>(first_moment.data());

        float *second_moment_data =
            static_cast<float *>(second_moment.data());

        const std::size_t count = value.numel();

        constexpr int block_size = 256;

        dim3 block(block_size);
        dim3 grid(static_cast<unsigned int>(
            (count + block_size - 1) / block_size));

        adam_update_cuda_float32_kernel<<<grid, block>>>(
            value_data,
            grad_data,
            first_moment_data,
            second_moment_data,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            beta1_power,
            beta2_power,
            count);

        check_cuda(
            cudaGetLastError(),
            "CUDA Adam update kernel launch failed");

        check_cuda(
            cudaDeviceSynchronize(),
            "CUDA Adam update synchronization failed");
    }

}