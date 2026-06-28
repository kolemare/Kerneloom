#include <kernels/rocm/conv2d/conv2d_rocm_float32_general.hiph>

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

        __global__ void conv2d_rocm_float32_general_kernel(
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
            std::size_t padding_h,
            std::size_t padding_w,
            std::size_t dilation_h,
            std::size_t dilation_w,
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

            for (std::size_t c = 0; c < C; ++c)
            {
                for (std::size_t r = 0; r < R; ++r)
                {
                    const long ih =
                        static_cast<long>(oh * stride_h) +
                        static_cast<long>(r * dilation_h) -
                        static_cast<long>(padding_h);

                    if (ih < 0 || ih >= static_cast<long>(H))
                    {
                        continue;
                    }

                    for (std::size_t s = 0; s < S; ++s)
                    {
                        const long iw =
                            static_cast<long>(ow * stride_w) +
                            static_cast<long>(s * dilation_w) -
                            static_cast<long>(padding_w);

                        if (iw < 0 || iw >= static_cast<long>(W))
                        {
                            continue;
                        }

                        const std::size_t input_index =
                            n * C * H * W +
                            c * H * W +
                            static_cast<std::size_t>(ih) * W +
                            static_cast<std::size_t>(iw);

                        const std::size_t kernel_index =
                            k * C * R * S +
                            c * R * S +
                            r * S +
                            s;

                        sum += input[input_index] * kernels[kernel_index];
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

    void conv2d_rocm_float32_general(
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
            conv2d_rocm_float32_general_kernel,
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
            options.padding_h,
            options.padding_w,
            options.dilation_h,
            options.dilation_w,
            use_bias);

        check_hip(hipGetLastError(), "ROCm conv2d general kernel launch failed");
    }

}