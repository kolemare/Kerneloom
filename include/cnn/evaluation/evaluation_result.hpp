#ifndef KL_EVALUATION_RESULT_HPP
#define KL_EVALUATION_RESULT_HPP

#include <cstddef>

namespace kl
{

    struct EvaluationResult
    {
        float average_loss = 0.0f;
        float accuracy = 0.0f;

        std::size_t batch_count = 0;
        std::size_t sample_count = 0;
        std::size_t correct_sample_count = 0;
    };

}

#endif // KL_EVALUATION_RESULT_HPP