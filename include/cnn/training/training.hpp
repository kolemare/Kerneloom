#ifndef KL_TRAINING_HPP
#define KL_TRAINING_HPP

#include <cnn/losses/loss.hpp>
#include <cnn/network/sequential.hpp>
#include <cnn/optimizers/optimizer.hpp>
#include <cnn/optimizers/parameter.hpp>
#include <cnn/training/training_result.hpp>

#include <core/tensor_pool.hpp>

#include <data/batch.hpp>
#include <data/data_loader.hpp>

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
            DataLoader &loader);

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