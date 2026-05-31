#include <data/dataset_split.hpp>

#include <algorithm>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

namespace kl
{

    DatasetSplit split_dataset(
        const std::vector<ImageSample> &samples,
        float train_fraction,
        float validation_fraction,
        std::uint32_t seed)
    {
        if (train_fraction < 0.0f ||
            validation_fraction < 0.0f ||
            train_fraction +
                    validation_fraction >
                1.0f)
        {
            throw std::runtime_error(
                "split_dataset received invalid fractions");
        }

        std::vector<ImageSample> shuffled =
            samples;

        std::mt19937 generator(
            seed);

        std::shuffle(
            shuffled.begin(),
            shuffled.end(),
            generator);

        const std::size_t train_count =
            static_cast<std::size_t>(
                shuffled.size() *
                train_fraction);

        const std::size_t validation_count =
            static_cast<std::size_t>(
                shuffled.size() *
                validation_fraction);

        const std::size_t validation_begin =
            train_count;

        const std::size_t test_begin =
            train_count +
            validation_count;

        DatasetSplit result;

        result.train.assign(
            shuffled.begin(),
            shuffled.begin() +
                validation_begin);

        result.validation.assign(
            shuffled.begin() +
                validation_begin,
            shuffled.begin() +
                test_begin);

        result.test.assign(
            shuffled.begin() +
                test_begin,
            shuffled.end());

        return result;
    }

}