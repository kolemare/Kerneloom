#include <cnn/training/training.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/tensor.hpp>

#include <stdexcept>

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
        model_.prepareTraining();

        model_.reset();
        loss_pool_.reset();

        Tensor &prediction =
            model_.forward(
                batch.inputs());

        Tensor &loss_value =
            loss_.forward(
                prediction,
                batch.targets(),
                loss_pool_);

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
            loss_data[0]};
    }

    TrainingEpochResult Training::trainEpoch(
        DataLoader &loader)
    {
        float total_loss =
            0.0f;

        std::size_t batch_count =
            0;

        while (loader.has_next())
        {
            Batch batch =
                loader.next();

            const TrainingResult result =
                trainBatch(
                    batch);

            total_loss +=
                result.loss;

            ++batch_count;
        }

        if (batch_count == 0)
        {
            throw std::runtime_error(
                "Training::trainEpoch received no batches");
        }

        return TrainingEpochResult{
            total_loss /
                static_cast<float>(
                    batch_count),
            batch_count};
    }

}