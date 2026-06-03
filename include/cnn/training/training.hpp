#ifndef KL_TRAINING_HPP
#define KL_TRAINING_HPP

#include <cnn/losses/loss.hpp>
#include <cnn/network/sequential.hpp>
#include <cnn/optimizers/optimizer.hpp>
#include <cnn/optimizers/parameter.hpp>
#include <cnn/training/training_progress.hpp>
#include <cnn/training/training_result.hpp>

#include <core/tensor_pool.hpp>

#include <data/batch.hpp>
#include <data/data_loader.hpp>

#include <cstddef>
#include <vector>

namespace kl
{

    class Training
    {
    public:
        Training(
            Sequential &model,
            Loss &loss,
            Optimizer &optimizer);

        TrainingResult trainBatch(
            Batch &batch);

        TrainingEpochResult trainEpoch(
            DataLoader &loader,
            std::size_t epoch = 1,
            std::size_t epoch_count = 1,
            const TrainingCallback &callback = {});

        std::vector<TrainingEpochResult> fit(
            DataLoader &loader,
            std::size_t epoch_count,
            const TrainingCallback &callback = {});

    private:
        Sequential &model_;
        Loss &loss_;
        Optimizer &optimizer_;

        std::vector<Parameter>
            parameters_;

        TensorPool loss_pool_;
    };

}

#endif // KL_TRAINING_HPP