#include <kernels/cpu/losses/categorical_cross_entropy_cpu_float32.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace kl
{

    void categorical_cross_entropy_cpu_float32(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &result,
        Reduction reduction,
        std::size_t valid_sample_count)
    {
        const std::size_t class_count =
            prediction.shape()[1];

        const float *prediction_data =
            static_cast<const float *>(
                prediction.data());

        const std::int32_t *target_data =
            static_cast<const std::int32_t *>(
                target.data());

        float *result_data =
            static_cast<float *>(
                result.data());

        constexpr float epsilon =
            1.0e-7f;

        float sum =
            0.0f;

        for (std::size_t n = 0;
             n < valid_sample_count;
             ++n)
        {
            const std::int32_t target_class =
                target_data[n];

            if (target_class < 0 ||
                target_class >=
                    static_cast<std::int32_t>(
                        class_count))
            {
                throw std::runtime_error(
                    "categorical_cross_entropy target class is out of range");
            }

            const std::size_t index =
                n *
                    class_count +
                static_cast<std::size_t>(
                    target_class);

            const float probability =
                std::clamp(
                    prediction_data[index],
                    epsilon,
                    1.0f -
                        epsilon);

            sum -=
                std::log(
                    probability);
        }

        if (reduction ==
            Reduction::Mean)
        {
            sum /=
                static_cast<float>(
                    valid_sample_count);
        }

        result_data[0] =
            sum;
    }

}