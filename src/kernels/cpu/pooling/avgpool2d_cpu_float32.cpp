#include <kernels/cpu/pooling/avgpool2d_cpu_float32.hpp>

#include <backend/cpu/cpu_parallel.hpp>

#include <cstddef>

namespace kl
{

    namespace
    {

        struct AvgPool2dCpuFloat32Task
        {
            const float *input_data;
            float *result_data;

            std::size_t N;
            std::size_t C;
            std::size_t H;
            std::size_t W;

            std::size_t OH;
            std::size_t OW;

            Pooling2dOptions options;

            void operator()(std::size_t begin, std::size_t end) const
            {
                for (std::size_t work = begin; work < end; ++work)
                {
                    const std::size_t oh = work % OH;
                    const std::size_t nc = work / OH;

                    const std::size_t n = nc / C;
                    const std::size_t c = nc % C;

                    const std::size_t input_nc_offset = n * C * H * W + c * H * W;
                    const std::size_t result_row_offset = n * C * OH * OW + c * OH * OW + oh * OW;

                    for (std::size_t ow = 0; ow < OW; ++ow)
                    {
                        float sum = 0.0f;
                        std::size_t count = 0;

                        for (std::size_t kh = 0; kh < options.kernel_h; ++kh)
                        {
                            const long ih =
                                static_cast<long>(oh * options.stride_h) +
                                static_cast<long>(kh) -
                                static_cast<long>(options.padding_h);

                            if (ih < 0 || ih >= static_cast<long>(H))
                            {
                                continue;
                            }

                            const std::size_t input_row_offset =
                                input_nc_offset + static_cast<std::size_t>(ih) * W;

                            for (std::size_t kw = 0; kw < options.kernel_w; ++kw)
                            {
                                const long iw =
                                    static_cast<long>(ow * options.stride_w) +
                                    static_cast<long>(kw) -
                                    static_cast<long>(options.padding_w);

                                if (iw < 0 || iw >= static_cast<long>(W))
                                {
                                    continue;
                                }

                                sum += input_data[input_row_offset + static_cast<std::size_t>(iw)];
                                ++count;
                            }
                        }

                        if (count > 0)
                        {
                            result_data[result_row_offset + ow] =
                                sum / static_cast<float>(count);
                        }
                        else
                        {
                            result_data[result_row_offset + ow] = 0.0f;
                        }
                    }
                }
            }
        };

    }

    void avgpool2d_cpu_float32(
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

        AvgPool2dCpuFloat32Task task{
            input_data,
            result_data,
            N,
            C,
            H,
            W,
            OH,
            OW,
            options};

        cpu_parallel_for(0, N * C * OH, task);
    }

}