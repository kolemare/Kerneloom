#include <kernels/cpu/optimizers/adam_update_cpu_float32.hpp>

#include <backend/cpu/cpu_parallel.hpp>

#include <cmath>
#include <cstddef>

namespace kl
{

    namespace
    {

        struct AdamUpdateCpuFloat32Task
        {
            float *value;
            const float *grad;
            float *first_moment;
            float *second_moment;

            float learning_rate;
            float beta1;
            float beta2;
            float epsilon;
            float beta1_power;
            float beta2_power;

            void operator()(std::size_t begin, std::size_t end) const
            {
                const float beta1_correction = 1.0f - beta1_power;
                const float beta2_correction = 1.0f - beta2_power;

                for (std::size_t i = begin; i < end; ++i)
                {
                    const float g = grad[i];

                    first_moment[i] =
                        beta1 * first_moment[i] +
                        (1.0f - beta1) * g;

                    second_moment[i] =
                        beta2 * second_moment[i] +
                        (1.0f - beta2) * g * g;

                    const float corrected_first =
                        first_moment[i] / beta1_correction;

                    const float corrected_second =
                        second_moment[i] / beta2_correction;

                    value[i] -=
                        learning_rate *
                        corrected_first /
                        (std::sqrt(corrected_second) + epsilon);
                }
            }
        };

    }

    void adam_update_cpu_float32(
        Tensor &value,
        const Tensor &grad,
        Tensor &first_moment,
        Tensor &second_moment,
        float learning_rate,
        float beta1,
        float beta2,
        float epsilon,
        float beta1_power,
        float beta2_power)
    {
        float *value_data =
            static_cast<float *>(value.data());

        const float *grad_data =
            static_cast<const float *>(grad.data());

        float *first_moment_data =
            static_cast<float *>(first_moment.data());

        float *second_moment_data =
            static_cast<float *>(second_moment.data());

        AdamUpdateCpuFloat32Task task{
            value_data,
            grad_data,
            first_moment_data,
            second_moment_data,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            beta1_power,
            beta2_power};

        cpu_parallel_for(
            0,
            value.numel(),
            task);
    }

}