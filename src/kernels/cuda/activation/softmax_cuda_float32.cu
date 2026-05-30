#include <kernels/cuda/activation/softmax_cuda_float32.cuh>

#include <cuda_runtime.h>

#include <cfloat>
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

        __global__ void softmax_cuda_float32_kernel(
            float *data,
            std::size_t class_count)
        {
            extern __shared__ float shared[];

            const std::size_t n =
                blockIdx.x;

            float *row =
                data +
                n * class_count;

            float local_maximum =
                -FLT_MAX;

            for (std::size_t c = threadIdx.x;
                 c < class_count;
                 c += blockDim.x)
            {
                if (row[c] > local_maximum)
                {
                    local_maximum = row[c];
                }
            }

            shared[threadIdx.x] =
                local_maximum;

            __syncthreads();

            for (unsigned int offset =
                     blockDim.x / 2;
                 offset > 0;
                 offset /= 2)
            {
                if (threadIdx.x < offset)
                {
                    if (shared[threadIdx.x + offset] >
                        shared[threadIdx.x])
                    {
                        shared[threadIdx.x] =
                            shared[threadIdx.x + offset];
                    }
                }

                __syncthreads();
            }

            const float maximum =
                shared[0];

            float local_sum =
                0.0f;

            for (std::size_t c = threadIdx.x;
                 c < class_count;
                 c += blockDim.x)
            {
                const float value =
                    expf(
                        row[c] -
                        maximum);

                row[c] = value;

                local_sum += value;
            }

            shared[threadIdx.x] =
                local_sum;

            __syncthreads();

            for (unsigned int offset =
                     blockDim.x / 2;
                 offset > 0;
                 offset /= 2)
            {
                if (threadIdx.x < offset)
                {
                    shared[threadIdx.x] +=
                        shared[threadIdx.x + offset];
                }

                __syncthreads();
            }

            const float sum =
                shared[0];

            for (std::size_t c = threadIdx.x;
                 c < class_count;
                 c += blockDim.x)
            {
                row[c] /= sum;
            }
        }

    }

    void softmax_cuda_float32(
        Tensor &tensor)
    {
        const std::size_t batch_size =
            tensor.shape()[0];

        const std::size_t class_count =
            tensor.shape()[1];

        float *data =
            static_cast<float *>(
                tensor.data());

        constexpr unsigned int block_size =
            256;

        const dim3 block(
            block_size);

        const dim3 grid(
            static_cast<unsigned int>(
                batch_size));

        const std::size_t shared_memory_size =
            block_size *
            sizeof(float);

        softmax_cuda_float32_kernel<<<
            grid,
            block,
            shared_memory_size>>>(
            data,
            class_count);

        check_cuda(
            cudaGetLastError(),
            "CUDA softmax kernel launch failed");

        check_cuda(
            cudaDeviceSynchronize(),
            "CUDA softmax synchronization failed");
    }

}