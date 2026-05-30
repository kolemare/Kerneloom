#include <kernels/cpu/activation/backward_softmax_cpu_float32.hpp>

#include <backend/cpu/cpu_parallel.hpp>

#include <cstddef>

namespace kl
{

    namespace
    {

        struct BackwardSoftmaxCpuFloat32Task
        {
            const float *activation_output;
            float *grad;

            std::size_t class_count;

            void operator()(
                std::size_t begin,
                std::size_t end) const
            {
                for (std::size_t n = begin; n < end; ++n)
                {
                    const float *output_row =
                        activation_output +
                        n * class_count;

                    float *grad_row =
                        grad +
                        n * class_count;

                    float dot = 0.0f;

                    for (std::size_t c = 0; c < class_count; ++c)
                    {
                        dot +=
                            grad_row[c] *
                            output_row[c];
                    }

                    for (std::size_t c = 0; c < class_count; ++c)
                    {
                        grad_row[c] =
                            output_row[c] *
                            (grad_row[c] - dot);
                    }
                }
            }
        };

    }

    void backward_softmax_cpu_float32(
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

        BackwardSoftmaxCpuFloat32Task task{
            activation_output_data,
            grad_data,
            class_count};

        cpu_parallel_for(
            0,
            batch_size,
            task);
    }

}