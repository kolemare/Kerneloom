#include <kernels/cpu/conv2d/conv2d_cpu_float32_valid.hpp>

#include <backend/cpu/cpu_parallel.hpp>

#include <cstddef>

namespace kl
{

    namespace
    {

        struct Conv2dCpuFloat32ValidTask
        {
            const float *input_data;
            const float *kernel_data;
            const float *bias_data;
            float *result_data;

            std::size_t N;
            std::size_t C;
            std::size_t H;
            std::size_t W;

            std::size_t K;
            std::size_t R;
            std::size_t S;

            std::size_t OH;
            std::size_t OW;

            bool use_bias;

            void operator()(std::size_t begin, std::size_t end) const
            {
                for (std::size_t work = begin; work < end; ++work)
                {
                    const std::size_t n = work / K;
                    const std::size_t k = work % K;

                    const std::size_t input_n_offset = n * C * H * W;
                    const std::size_t kernel_k_offset = k * C * R * S;
                    const std::size_t result_nk_offset = n * K * OH * OW + k * OH * OW;

                    for (std::size_t oh = 0; oh < OH; ++oh)
                    {
                        for (std::size_t ow = 0; ow < OW; ++ow)
                        {
                            float sum = 0.0f;

                            if (use_bias && bias_data != nullptr)
                            {
                                sum = bias_data[k];
                            }

                            for (std::size_t c = 0; c < C; ++c)
                            {
                                const std::size_t input_c_offset = input_n_offset + c * H * W;
                                const std::size_t kernel_c_offset = kernel_k_offset + c * R * S;

                                for (std::size_t r = 0; r < R; ++r)
                                {
                                    const std::size_t input_row_offset =
                                        input_c_offset + (oh + r) * W;

                                    const std::size_t kernel_row_offset =
                                        kernel_c_offset + r * S;

                                    for (std::size_t s = 0; s < S; ++s)
                                    {
                                        sum += input_data[input_row_offset + ow + s] *
                                               kernel_data[kernel_row_offset + s];
                                    }
                                }
                            }

                            result_data[result_nk_offset + oh * OW + ow] = sum;
                        }
                    }
                }
            }
        };

    }

    void conv2d_cpu_float32_valid(
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

        Conv2dCpuFloat32ValidTask task{
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
            options.use_bias};

        cpu_parallel_for(0, N * K, task);
    }

}