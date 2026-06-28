#include <kernels/rocm/activation/backward_softmax_rocm_float32.hiph>

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
            if (error != hipSuccess)
            {
                throw std::runtime_error(
                    std::string(message) +
                    ": " +
                    hipGetErrorString(error));
            }
        }

        __global__ void backward_softmax_rocm_float32_kernel(
            const float *activation_output,
            float *grad,
            std::size_t class_count)
        {
            extern __shared__ float shared[];

            const std::size_t n =
                blockIdx.x;

            const float *output_row =
                activation_output +
                n * class_count;

            float *grad_row =
                grad +
                n * class_count;

            float local_dot =
                0.0f;

            for (std::size_t c = threadIdx.x;
                 c < class_count;
                 c += blockDim.x)
            {
                local_dot +=
                    grad_row[c] *
                    output_row[c];
            }

            shared[threadIdx.x] =
                local_dot;

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

            const float dot =
                shared[0];

            for (std::size_t c = threadIdx.x;
                 c < class_count;
                 c += blockDim.x)
            {
                grad_row[c] =
                    output_row[c] *
                    (grad_row[c] - dot);
            }
        }

    }

    void backward_softmax_rocm_float32(
        const Tensor &activation_output,
        Tensor &grad)
    {
        const std::size_t batch_size =
            activation_output.shape()[0];

        const std::size_t class_count =
            activation_output.shape()[1];

        const float *activation_output_data =
            static_cast<const float *>(
                activation_output.data());

        float *grad_data =
            static_cast<float *>(
                grad.data());

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

        hipLaunchKernelGGL(
            backward_softmax_rocm_float32_kernel,
            grid,
            block,
            shared_memory_size,
            0,
            activation_output_data,
            grad_data,
            class_count);

        check_hip(
            hipGetLastError(),
            "ROCm backward softmax kernel launch failed");
    }

}