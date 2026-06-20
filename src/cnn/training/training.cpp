#include <cnn/training/training.hpp>

#include <cnn/metrics/classification_metrics.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/tensor.hpp>

#include <stdexcept>
#include <vector>

namespace kl
{

    Training::Training(
        Sequential &model,
        Loss &loss,
        Optimizer &optimizer)
        : model_(
              model),
          loss_(
              loss),
          optimizer_(
              optimizer)
    {
        model_.prepareTraining();

        model_.collectParameters(
            parameters_);

        optimizer_.prepare(
            parameters_);
    }

    TrainingResult Training::trainBatch(
        Batch &batch)
    {
        model_.setMode(
            LayerMode::Training);

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

        const std::size_t correct_sample_count =
            count_correct_predictions(
                prediction,
                batch.targets(),
                batch.valid_sample_count());

        const float accuracy =
            static_cast<float>(
                correct_sample_count) /
            static_cast<float>(
                batch.valid_sample_count());

        Tensor &grad_prediction =
            loss_.backward(
                loss_pool_);

        model_.backward(
            grad_prediction);

        optimizer_.step(
            parameters_);

        Tensor loss_cpu =
            loss_value.to(
                Device::cpu());

        if (loss_cpu.dtype() !=
                DType::Float32 ||
            loss_cpu.numel() !=
                1)
        {
            throw std::runtime_error(
                "Training expects scalar Float32 loss");
        }

        const float *loss_data =
            static_cast<const float *>(
                loss_cpu.data());

        return TrainingResult{
            loss_data[0],
            accuracy,
            batch.valid_sample_count(),
            correct_sample_count};
    }

    TrainingEpochResult Training::trainEpoch(
        DataLoader &loader,
        std::size_t epoch,
        std::size_t epoch_count,
        const TrainingCallback &callback)
    {
        float total_weighted_loss =
            0.0f;

        std::size_t batch_index =
            0;

        std::size_t total_sample_count =
            0;

        std::size_t total_correct_sample_count =
            0;

        const std::size_t batch_count =
            loader.batch_count();

        while (loader.has_next())
        {
            Batch batch =
                loader.next();

            const TrainingResult result =
                trainBatch(
                    batch);

            ++batch_index;

            total_weighted_loss +=
                result.loss *
                static_cast<float>(
                    result.sample_count);

            total_sample_count +=
                result.sample_count;

            total_correct_sample_count +=
                result.correct_sample_count;

            const float average_loss =
                total_weighted_loss /
                static_cast<float>(
                    total_sample_count);

            const float average_accuracy =
                static_cast<float>(
                    total_correct_sample_count) /
                static_cast<float>(
                    total_sample_count);

            if (callback)
            {
                callback(
                    TrainingProgress{
                        epoch,
                        epoch_count,
                        batch_index,
                        batch_count,
                        result.loss,
                        average_loss,
                        result.accuracy,
                        average_accuracy,
                        false});
            }
        }

        if (batch_index == 0)
        {
            throw std::runtime_error(
                "Training::trainEpoch received no batches");
        }

        if (total_sample_count == 0)
        {
            throw std::runtime_error(
                "Training::trainEpoch received no valid samples");
        }

        const TrainingEpochResult result{
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

        if (callback)
        {
            callback(
                TrainingProgress{
                    epoch,
                    epoch_count,
                    batch_index,
                    batch_count,
                    result.average_loss,
                    result.average_loss,
                    result.accuracy,
                    result.accuracy,
                    true});
        }

        return result;
    }

    std::vector<TrainingEpochResult>
    Training::fit(
        DataLoader &loader,
        std::size_t epoch_count,
        const TrainingCallback &callback)
    {
        if (epoch_count == 0)
        {
            throw std::runtime_error(
                "Training::fit epoch count must be greater than zero");
        }

        std::vector<TrainingEpochResult>
            results;

        results.reserve(
            epoch_count);

        for (std::size_t epoch = 1;
             epoch <= epoch_count;
             ++epoch)
        {
            if (epoch > 1)
            {
                loader.reset_epoch();
            }

            results.push_back(
                trainEpoch(
                    loader,
                    epoch,
                    epoch_count,
                    callback));
        }

        return results;
    }

}