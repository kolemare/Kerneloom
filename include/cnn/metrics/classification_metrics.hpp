#ifndef KL_CLASSIFICATION_METRICS_HPP
#define KL_CLASSIFICATION_METRICS_HPP

#include <core/tensor.hpp>

#include <cstddef>

namespace kl
{

    std::size_t count_correct_predictions(
        const Tensor &predictions,
        const Tensor &targets,
        std::size_t valid_sample_count);

    float classification_accuracy(
        const Tensor &predictions,
        const Tensor &targets,
        std::size_t valid_sample_count);

}

#endif // KL_CLASSIFICATION_METRICS_HPP