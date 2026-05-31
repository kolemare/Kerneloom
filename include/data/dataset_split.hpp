#ifndef KL_DATASET_SPLIT_HPP
#define KL_DATASET_SPLIT_HPP

#include <data/image_sample.hpp>

#include <cstdint>
#include <vector>

namespace kl
{

    struct DatasetSplit
    {
        std::vector<ImageSample> train;
        std::vector<ImageSample> validation;
        std::vector<ImageSample> test;
    };

    DatasetSplit split_dataset(
        const std::vector<ImageSample> &samples,
        float train_fraction,
        float validation_fraction,
        std::uint32_t seed = 1337);

}

#endif // KL_DATASET_SPLIT_HPP