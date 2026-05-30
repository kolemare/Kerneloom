#include <kernels/cpu/losses/backward_mse_loss_cpu_float32.hpp>

#include <backend/cpu/cpu_parallel.hpp>

#include <cstddef>

namespace kl
{

    namespace
    {

        struct BackwardMSELossCpuFloat32Task
        {
            const float *prediction;
            const float *target;
            float *grad_prediction;

            float scale;

            void operator()(
                std::size_t begin,
                std::size_t end) const
            {
                for (std::size_t i = begin; i < end; ++i)
                {
                    grad_prediction[i] =
                        scale *
                        (prediction[i] - target[i]);
                }
            }
        };

    }

    void backward_mse_loss_cpu_float32(
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

        float scale = 2.0f;

        if (reduction == Reduction::Mean)
        {
            scale /=
                static_cast<float>(prediction.numel());
        }

        BackwardMSELossCpuFloat32Task task{
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