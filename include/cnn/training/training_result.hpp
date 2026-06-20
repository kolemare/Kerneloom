#ifndef KL_TRAINING_RESULT_HPP
#define KL_TRAINING_RESULT_HPP

#include <cstddef>

namespace kl
{

    struct TrainingResult
    {
        float loss = 0.0f;
        float accuracy = 0.0f;

        std::size_t sample_count = 0;
        std::size_t correct_sample_count = 0;
    };

    struct TrainingEpochResult
    {
        float average_loss = 0.0f;
        float accuracy = 0.0f;

        std::size_t batch_count = 0;
        std::size_t sample_count = 0;
        std::size_t correct_sample_count = 0;
    };

}

#endif // KL_TRAINING_RESULT_HPP