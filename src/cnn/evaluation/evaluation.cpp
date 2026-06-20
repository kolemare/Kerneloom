#include <cnn/evaluation/evaluation.hpp>

#include <cnn/metrics/classification_metrics.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/tensor.hpp>

#include <data/batch.hpp>

#include <stdexcept>

namespace kl
{

    namespace
    {

        float scalar_loss_value(
            const Tensor &loss_value)
        {
            Tensor loss_cpu =
                loss_value.to(
                    Device::cpu());

            if (loss_cpu.dtype() !=
                    DType::Float32 ||
                loss_cpu.numel() !=
                    1)
            {
                throw std::runtime_error(
                    "Evaluation expects scalar Float32 loss");
            }

            const float *loss_data =
                static_cast<const float *>(
                    loss_cpu.data());

            return loss_data[0];
        }

    }

    Evaluation::Evaluation(
        Sequential &model,
        Loss &loss)
        : model_(
              model),
          loss_(
              loss)
    {
        model_.prepareInference();
    }

    EvaluationResult Evaluation::evaluate(
        DataLoader &loader)
    {
        model_.prepareInference();

        float total_weighted_loss =
            0.0f;

        std::size_t batch_index =
            0;

        std::size_t total_sample_count =
            0;

        std::size_t total_correct_sample_count =
            0;

        while (loader.has_next())
        {
            Batch batch =
                loader.next();

            model_.reset();
            loss_pool_.reset();

            Tensor &prediction =
                model_.forward(
                    batch.inputs());

            Tensor &loss_value =
                loss_.forward(
                    prediction,
                    batch.targets(),
                    loss_pool_,
                    batch.valid_sample_count());

            const float batch_loss =
                scalar_loss_value(
                    loss_value);

            const std::size_t correct_sample_count =
                count_correct_predictions(
                    prediction,
                    batch.targets(),
                    batch.valid_sample_count());

            ++batch_index;

            total_weighted_loss +=
                batch_loss *
                static_cast<float>(
                    batch.valid_sample_count());

            total_sample_count +=
                batch.valid_sample_count();

            total_correct_sample_count +=
                correct_sample_count;
        }

        if (batch_index == 0)
        {
            throw std::runtime_error(
                "Evaluation::evaluate received no batches");
        }

        if (total_sample_count == 0)
        {
            throw std::runtime_error(
                "Evaluation::evaluate received no valid samples");
        }

        return EvaluationResult{
            total_weighted_loss /
                static_cast<float>(
                    total_sample_count),
            static_cast<float>(
                total_correct_sample_count) /
                static_cast<float>(
                    total_sample_count),
            batch_index,
            total_sample_count,
            total_correct_sample_count};
    }

}