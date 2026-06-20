#include <cnn/metrics/classification_metrics.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>

#include <cstdint>
#include <stdexcept>

namespace kl
{

    std::size_t count_correct_predictions(
        const Tensor &predictions,
        const Tensor &targets,
        std::size_t valid_sample_count)
    {
        if (predictions.rank() != 2)
        {
            throw std::runtime_error(
                "count_correct_predictions expects predictions with rank 2");
        }

        if (targets.rank() != 1)
        {
            throw std::runtime_error(
                "count_correct_predictions expects targets with rank 1");
        }

        if (predictions.dtype() != DType::Float32)
        {
            throw std::runtime_error(
                "count_correct_predictions expects Float32 predictions");
        }

        if (targets.dtype() != DType::Int32)
        {
            throw std::runtime_error(
                "count_correct_predictions expects Int32 targets");
        }

        if (predictions.shape()[0] != targets.shape()[0])
        {
            throw std::runtime_error(
                "count_correct_predictions expects matching batch dimensions");
        }

        if (valid_sample_count > predictions.shape()[0])
        {
            throw std::runtime_error(
                "count_correct_predictions received invalid sample count");
        }

        const std::size_t class_count =
            predictions.shape()[1];

        if (class_count == 0)
        {
            throw std::runtime_error(
                "count_correct_predictions received zero classes");
        }

        Tensor predictions_cpu =
            predictions.to(
                Device::cpu());

        Tensor targets_cpu =
            targets.to(
                Device::cpu());

        const float *prediction_data =
            static_cast<const float *>(
                predictions_cpu.data());

        const std::int32_t *target_data =
            static_cast<const std::int32_t *>(
                targets_cpu.data());

        std::size_t correct_count =
            0;

        for (std::size_t sample = 0;
             sample < valid_sample_count;
             ++sample)
        {
            const float *row =
                prediction_data +
                sample *
                    class_count;

            std::size_t predicted_class =
                0;

            float best_value =
                row[0];

            for (std::size_t class_index = 1;
                 class_index < class_count;
                 ++class_index)
            {
                if (row[class_index] > best_value)
                {
                    best_value =
                        row[class_index];

                    predicted_class =
                        class_index;
                }
            }

            const std::int32_t target_class =
                target_data[sample];

            if (target_class < 0 ||
                static_cast<std::size_t>(target_class) >= class_count)
            {
                throw std::runtime_error(
                    "count_correct_predictions received target class out of range");
            }

            if (predicted_class ==
                static_cast<std::size_t>(target_class))
            {
                ++correct_count;
            }
        }

        return correct_count;
    }

    float classification_accuracy(
        const Tensor &predictions,
        const Tensor &targets,
        std::size_t valid_sample_count)
    {
        if (valid_sample_count == 0)
        {
            throw std::runtime_error(
                "classification_accuracy received zero valid samples");
        }

        const std::size_t correct_count =
            count_correct_predictions(
                predictions,
                targets,
                valid_sample_count);

        return static_cast<float>(correct_count) /
               static_cast<float>(valid_sample_count);
    }

}