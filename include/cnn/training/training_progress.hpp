#ifndef KL_TRAINING_PROGRESS_HPP
#define KL_TRAINING_PROGRESS_HPP

#include <cstddef>
#include <functional>

namespace kl
{

    struct TrainingProgress
    {
        std::size_t epoch =
            0;

        std::size_t epoch_count =
            0;

        std::size_t batch =
            0;

        std::size_t batch_count =
            0;

        float batch_loss =
            0.0f;

        float average_loss =
            0.0f;

        float batch_accuracy =
            0.0f;

        float average_accuracy =
            0.0f;

        bool epoch_complete =
            false;
    };

    using TrainingCallback =
        std::function<void(
            const TrainingProgress &progress)>;

}

#endif // KL_TRAINING_PROGRESS_HPP