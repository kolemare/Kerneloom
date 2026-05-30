#include <kernels/cpu/losses/mse_loss_cpu_float32.hpp>

#include <cstddef>

namespace kl
{

    void mse_loss_cpu_float32(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &result,
        Reduction reduction)
    {
        const float *prediction_data =
            static_cast<const float *>(prediction.data());

        const float *target_data =
            static_cast<const float *>(target.data());

        float *result_data =
            static_cast<float *>(result.data());

        float sum = 0.0f;

        for (std::size_t i = 0; i < prediction.numel(); ++i)
        {
            const float difference =
                prediction_data[i] - target_data[i];

            sum += difference * difference;
        }

        if (reduction == Reduction::Mean)
        {
            sum /= static_cast<float>(prediction.numel());
        }

        result_data[0] = sum;
    }

}