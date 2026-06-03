#ifndef KL_TRAINING_CALLBACKS_HPP
#define KL_TRAINING_CALLBACKS_HPP

#include <cnn/training/training_progress.hpp>

namespace kl
{

    namespace training_callbacks
    {

        TrainingCallback terminal_loss();

        TrainingCallback epoch_loss();

    }

}

#endif // KL_TRAINING_CALLBACKS_HPP