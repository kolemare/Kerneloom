#include <kernels/cpu/conv2d_cpu_float32_naive.hpp>

namespace kl
{
    // output
    // N x K x OH x OW

    void conv2d_cpu_float32_naive(
        const Tensor &input,
        const Tensor &kernels,
        Tensor &result)
    {
        const std::size_t N = input.shape()[0];
        const std::size_t C = input.shape()[1];
        const std::size_t H = input.shape()[2];
        const std::size_t W = input.shape()[3];

        const std::size_t K = kernels.shape()[0];
        // kernel channels are same as input
        const std::size_t R = kernels.shape()[2];
        const std::size_t S = kernels.shape()[3];

        const std::size_t OH = input.shape()[2] - kernels.shape()[2] + 1;
        const std::size_t OW = input.shape()[3] - kernels.shape()[3] + 1;

        const float *input_data = static_cast<const float *>(input.data());
        const float *kernels_data = static_cast<const float *>(kernels.data());
        float *result_data = static_cast<float *>(result.data());

        for (int n = 0; n < N; n++)
        {
            for (int k = 0; k < K; k++)
            {
                for (int oh = 0; oh < OH; oh++)
                {
                    for (int ow = 0; ow < OW; ow++)
                    {
                        float sum = 0;
                        for (int c = 0; c < C; c++)
                        {
                            for (int r = 0; r < R; r++)
                            {
                                for (int s = 0; s < S; s++)
                                {

                                    sum += input_data[n * C * H * W + c * H * W + (oh + r) * W + (ow + s)] * kernels_data[k * C * R * S + c * R * S + r * S + s];
                                }
                            }
                        }
                        result_data[n * K * OH * OW + k * OH * OW + oh * OW + ow] = sum;
                    }
                }
            }
        }
    }
}