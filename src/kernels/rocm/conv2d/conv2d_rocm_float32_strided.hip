#include <kernels/rocm/conv2d/conv2d_rocm_float32_strided.hiph>

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

namespace kl
{
    namespace
    {

        void check_hip(hipError_t error, const char *message)
        {
            if (error != hipSuccess)
            {
                throw std::runtime_error(
                    std::string(message) + ": " + hipGetErrorString(error));
            }
        }

        __global__ void conv2d_rocm_float32_strided_kernel(
            const float *input,
            const float *kernels,
            const float *bias,
            float *result,
            std::size_t N,
            std::size_t C,
            std::size_t H,
            std::size_t W,
            std::size_t K,
            std::size_t R,
            std::size_t S,
            std::size_t OH,
            std::size_t OW,
            std::size_t stride_h,
            std::size_t stride_w,
            bool use_bias)
        {
            const std::size_t ow = blockIdx.x * blockDim.x + threadIdx.x;
            const std::size_t oh = blockIdx.y * blockDim.y + threadIdx.y;

            const std::size_t nk = blockIdx.z;
            const std::size_t n = nk / K;
            const std::size_t k = nk % K;

            if (oh >= OH || ow >= OW)
            {
                return;
            }

            float sum = 0.0f;

            if (use_bias && bias != nullptr)
            {
                sum = bias[k];
            }

            const std::size_t input_n_offset = n * C * H * W;
            const std::size_t kernel_k_offset = k * C * R * S;

            const std::size_t input_h_base = oh * stride_h;
            const std::size_t input_w_base = ow * stride_w;

            for (std::size_t c = 0; c < C; ++c)
            {
                const std::size_t input_c_offset = input_n_offset + c * H * W;
                const std::size_t kernel_c_offset = kernel_k_offset + c * R * S;

                for (std::size_t r = 0; r < R; ++r)
                {
                    const std::size_t input_row_offset =
                        input_c_offset + (input_h_base + r) * W;

                    const std::size_t kernel_row_offset =
                        kernel_c_offset + r * S;

                    for (std::size_t s = 0; s < S; ++s)
                    {
                        sum += input[input_row_offset + input_w_base + s] *
                               kernels[kernel_row_offset + s];
                    }
                }
            }

            const std::size_t result_index =
                n * K * OH * OW +
                k * OH * OW +
                oh * OW +
                ow;

            result[result_index] = sum;
        }

    }

    void conv2d_rocm_float32_strided(
        const Tensor &input,
        const Tensor &kernels,
        const Tensor *bias,
        Tensor &result,
        const Conv2dOptions &options)
    {
        const std::size_t N = input.shape()[0];
        const std::size_t C = input.shape()[1];
        const std::size_t H = input.shape()[2];
        const std::size_t W = input.shape()[3];

        const std::size_t K = kernels.shape()[0];
        const std::size_t R = kernels.shape()[2];
        const std::size_t S = kernels.shape()[3];

        const std::size_t OH = result.shape()[2];
        const std::size_t OW = result.shape()[3];

        const float *input_data = static_cast<const float *>(input.data());
        const float *kernel_data = static_cast<const float *>(kernels.data());

        const float *bias_data = nullptr;

        if (bias != nullptr)
        {
            bias_data = static_cast<const float *>(bias->data());
        }

        float *result_data = static_cast<float *>(result.data());

        const bool use_bias = options.use_bias && bias_data != nullptr;

        constexpr int block_size = 16;

        dim3 block(block_size, block_size);
        dim3 grid(
            static_cast<unsigned int>((OW + block_size - 1) / block_size),
            static_cast<unsigned int>((OH + block_size - 1) / block_size),
            static_cast<unsigned int>(N * K));

        hipLaunchKernelGGL(
            conv2d_rocm_float32_strided_kernel,
            grid,
            block,
            0,
            0,
            input_data,
            kernel_data,
            bias_data,
            result_data,
            N,
            C,
            H,
            W,
            K,
            R,
            S,
            OH,
            OW,
            options.stride_h,
            options.stride_w,
            use_bias);

        check_hip(hipGetLastError(), "ROCm conv2d strided kernel launch failed");
        check_hip(hipDeviceSynchronize(), "ROCm conv2d strided synchronization failed");
    }

}