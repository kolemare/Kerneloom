#include <kernels/cpu/pooling/backward_maxpool2d_cpu_float32.hpp>

#include <backend/cpu/cpu_parallel.hpp>

#include <cstddef>
#include <cstdint>

namespace kl
{

    namespace
    {

        struct ZeroGradInputTask
        {
            float *grad_input;

            void operator()(std::size_t begin, std::size_t end) const
            {
                for (std::size_t i = begin; i < end; ++i)
                {
                    grad_input[i] = 0.0f;
                }
            }
        };

    }

    void backward_maxpool2d_cpu_float32(
        const Tensor &indices,
        const Tensor &grad_output,
        Tensor &grad_input)
    {
        const std::int32_t *indices_data =
            static_cast<const std::int32_t *>(indices.data());

        const float *grad_output_data =
            static_cast<const float *>(grad_output.data());

        float *grad_input_data =
            static_cast<float *>(grad_input.data());

        ZeroGradInputTask zero_task{
            grad_input_data};

        cpu_parallel_for(
            0,
            grad_input.numel(),
            zero_task);

        for (std::size_t i = 0; i < grad_output.numel(); ++i)
        {
            const std::int32_t index = indices_data[i];

            grad_input_data[index] += grad_output_data[i];
        }
    }

}