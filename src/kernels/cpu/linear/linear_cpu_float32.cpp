#include <kernels/cpu/linear/linear_cpu_float32.hpp>

#include <backend/cpu/cpu_parallel.hpp>

#include <cstddef>

namespace kl
{

    namespace
    {

        struct LinearCpuFloat32Task
        {
            const float *input;
            const float *weights;
            const float *bias;
            float *result;

            std::size_t batch_size;
            std::size_t input_features;
            std::size_t output_features;

            void operator()(std::size_t begin, std::size_t end) const
            {
                for (std::size_t work = begin; work < end; ++work)
                {
                    const std::size_t n = work / output_features;
                    const std::size_t out = work % output_features;

                    float sum = 0.0f;

                    const std::size_t input_offset = n * input_features;
                    const std::size_t weight_offset = out * input_features;

                    for (std::size_t in = 0; in < input_features; ++in)
                    {
                        sum += input[input_offset + in] *
                               weights[weight_offset + in];
                    }

                    if (bias != nullptr)
                    {
                        sum += bias[out];
                    }

                    result[work] = sum;
                }
            }
        };

    }

    void linear_cpu_float32(
        const Tensor &input,
        const Tensor &weights,
        const Tensor *bias,
        Tensor &result)
    {
        const std::size_t batch_size = input.shape()[0];
        const std::size_t input_features = input.shape()[1];
        const std::size_t output_features = weights.shape()[0];

        const float *input_data = static_cast<const float *>(input.data());
        const float *weights_data = static_cast<const float *>(weights.data());

        const float *bias_data = nullptr;

        if (bias != nullptr)
        {
            bias_data = static_cast<const float *>(bias->data());
        }

        float *result_data = static_cast<float *>(result.data());

        LinearCpuFloat32Task task{
            input_data,
            weights_data,
            bias_data,
            result_data,
            batch_size,
            input_features,
            output_features};

        cpu_parallel_for(
            0,
            batch_size * output_features,
            task);
    }

}