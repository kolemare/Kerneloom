#include <kernels/rocm/pooling/maxpool2d_rocm_float32.hiph>

#include <hip/hip_runtime.h>

#include <cfloat>
#include <cstddef>
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

        __global__ void maxpool2d_rocm_float32_kernel(
            const float *input,
            float *result,
            std::size_t N,
            std::size_t C,
            std::size_t H,
            std::size_t W,
            std::size_t OH,
            std::size_t OW,
            std::size_t kernel_h,
            std::size_t kernel_w,
            std::size_t stride_h,
            std::size_t stride_w,
            std::size_t padding_h,
            std::size_t padding_w)
        {
            const std::size_t ow = blockIdx.x * blockDim.x + threadIdx.x;
            const std::size_t oh = blockIdx.y * blockDim.y + threadIdx.y;

            const std::size_t nc = blockIdx.z;
            const std::size_t n = nc / C;
            const std::size_t c = nc % C;

            if (oh >= OH || ow >= OW)
            {
                return;
            }

            float max_value = -FLT_MAX;
            bool has_value = false;

            const std::size_t input_nc_offset =
                n * C * H * W +
                c * H * W;

            for (std::size_t kh = 0; kh < kernel_h; ++kh)
            {
                const long ih =
                    static_cast<long>(oh * stride_h) +
                    static_cast<long>(kh) -
                    static_cast<long>(padding_h);

                if (ih < 0 || ih >= static_cast<long>(H))
                {
                    continue;
                }

                const std::size_t input_row_offset =
                    input_nc_offset + static_cast<std::size_t>(ih) * W;

                for (std::size_t kw = 0; kw < kernel_w; ++kw)
                {
                    const long iw =
                        static_cast<long>(ow * stride_w) +
                        static_cast<long>(kw) -
                        static_cast<long>(padding_w);

                    if (iw < 0 || iw >= static_cast<long>(W))
                    {
                        continue;
                    }

                    const float value =
                        input[input_row_offset + static_cast<std::size_t>(iw)];

                    if (!has_value || value > max_value)
                    {
                        max_value = value;
                        has_value = true;
                    }
                }
            }

            const std::size_t result_index =
                n * C * OH * OW +
                c * OH * OW +
                oh * OW +
                ow;

            result[result_index] = has_value ? max_value : 0.0f;
        }

    }

    void maxpool2d_rocm_float32(
        const Tensor &input,
        Tensor &result,
        const Pooling2dOptions &options)
    {
        const std::size_t N = input.shape()[0];
        const std::size_t C = input.shape()[1];
        const std::size_t H = input.shape()[2];
        const std::size_t W = input.shape()[3];

        const std::size_t OH = result.shape()[2];
        const std::size_t OW = result.shape()[3];

        const float *input_data = static_cast<const float *>(input.data());
        float *result_data = static_cast<float *>(result.data());

        constexpr int block_size = 16;

        dim3 block(block_size, block_size);
        dim3 grid(
            static_cast<unsigned int>((OW + block_size - 1) / block_size),
            static_cast<unsigned int>((OH + block_size - 1) / block_size),
            static_cast<unsigned int>(N * C));

        hipLaunchKernelGGL(
            maxpool2d_rocm_float32_kernel,
            grid,
            block,
            0,
            0,
            input_data,
            result_data,
            N,
            C,
            H,
            W,
            OH,
            OW,
            options.kernel_h,
            options.kernel_w,
            options.stride_h,
            options.stride_w,
            options.padding_h,
            options.padding_w);

        check_hip(hipGetLastError(), "ROCm maxpool2d kernel launch failed");
    }

}