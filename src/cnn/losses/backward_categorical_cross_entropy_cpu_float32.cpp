#include <kernels/cpu/losses/backward_categorical_cross_entropy_cpu_float32.hpp>

#include <backend/cpu/cpu_parallel.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace kl
{

    namespace
    {

        struct ZeroGradientTask
        {
            float *grad_prediction;

            void operator()(
                std::size_t begin,
                std::size_t end) const
            {
                for (std::size_t i = begin; i < end; ++i)
                {
                    grad_prediction[i] = 0.0f;
                }
            }
        };

    }

    void backward_categorical_cross_entropy_cpu_float32(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &grad_prediction,
        Reduction reduction)
    {
        const std::size_t batch_size =
            prediction.shape()[0];

        const std::size_t class_count =
            prediction.shape()[1];

        const float *prediction_data =
            static_cast<const float *>(prediction.data());

        const std::int32_t *target_data =
            static_cast<const std::int32_t *>(target.data());

        float *grad_prediction_data =
            static_cast<float *>(grad_prediction.data());

        ZeroGradientTask zero_task{
            grad_prediction_data};

        cpu_parallel_for(
            0,
            grad_prediction.numel(),
            zero_task);

        constexpr float epsilon = 1.0e-7f;

        float scale = 1.0f;

        if (reduction == Reduction::Mean)
        {
            scale /=
                static_cast<float>(batch_size);
        }

        for (std::size_t n = 0; n < batch_size; ++n)
        {
            const std::int32_t target_class =
                target_data[n];

            if (target_class < 0 ||
                target_class >= static_cast<std::int32_t>(class_count))
            {
                throw std::runtime_error(
                    "backward_categorical_cross_entropy target class is out of range");
            }

            const std::size_t index =
                n * class_count +
                static_cast<std::size_t>(target_class);

            const float probability =
                std::clamp(
                    prediction_data[index],
                    epsilon,
                    1.0f - epsilon);

            grad_prediction_data[index] =
                -scale / probability;
        }
    }

}