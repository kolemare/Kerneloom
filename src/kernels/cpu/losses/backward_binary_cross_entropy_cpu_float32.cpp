#include <kernels/cpu/losses/backward_binary_cross_entropy_cpu_float32.hpp>

#include <backend/cpu/cpu_parallel.hpp>

#include <algorithm>
#include <cstddef>

namespace kl
{

    namespace
    {

        struct BackwardBinaryCrossEntropyCpuFloat32Task
        {
            const float *prediction;
            const float *target;
            float *grad_prediction;

            float scale;

            void operator()(
                std::size_t begin,
                std::size_t end) const
            {
                constexpr float epsilon = 1.0e-7f;

                for (std::size_t i = begin; i < end; ++i)
                {
                    const float probability =
                        std::clamp(
                            prediction[i],
                            epsilon,
                            1.0f - epsilon);

                    grad_prediction[i] =
                        scale *
                        (-target[i] / probability +
                         (1.0f - target[i]) /
                             (1.0f - probability));
                }
            }
        };

    }

    void backward_binary_cross_entropy_cpu_float32(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &grad_prediction,
        Reduction reduction)
    {
        const float *prediction_data =
            static_cast<const float *>(prediction.data());

        const float *target_data =
            static_cast<const float *>(target.data());

        float *grad_prediction_data =
            static_cast<float *>(grad_prediction.data());

        float scale = 1.0f;

        if (reduction == Reduction::Mean)
        {
            scale /=
                static_cast<float>(prediction.numel());
        }

        BackwardBinaryCrossEntropyCpuFloat32Task task{
            prediction_data,
            target_data,
            grad_prediction_data,
            scale};

        cpu_parallel_for(
            0,
            prediction.numel(),
            task);
    }

}