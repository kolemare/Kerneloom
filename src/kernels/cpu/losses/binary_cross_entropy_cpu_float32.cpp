#include <kernels/cpu/losses/binary_cross_entropy_cpu_float32.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace kl
{

    void binary_cross_entropy_cpu_float32(
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

        constexpr float epsilon = 1.0e-7f;

        float sum = 0.0f;

        for (std::size_t i = 0; i < prediction.numel(); ++i)
        {
            const float probability =
                std::clamp(
                    prediction_data[i],
                    epsilon,
                    1.0f - epsilon);

            const float target_value =
                target_data[i];

            sum -=
                target_value *
                    std::log(probability) +
                (1.0f - target_value) *
                    std::log(1.0f - probability);
        }

        if (reduction == Reduction::Mean)
        {
            sum /=
                static_cast<float>(prediction.numel());
        }

        result_data[0] = sum;
    }

}