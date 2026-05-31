#ifndef KL_DATA_LOADER_HPP
#define KL_DATA_LOADER_HPP

#include <data/batch.hpp>
#include <data/data_loader_options.hpp>
#include <data/image_decoder.hpp>
#include <data/image_sample.hpp>
#include <data/image_transform.hpp>

#include <core/device.hpp>

#include <cstddef>
#include <vector>

namespace kl
{

    class DataLoader
    {
    public:
        DataLoader(
            std::vector<ImageSample> samples,
            ImageTransform transform,
            Device device,
            DataLoaderOptions options = {});

        void reset_epoch();

        bool has_next() const;

        Batch next();

        std::size_t sample_count() const;
        std::size_t batch_count() const;

    private:
        std::vector<ImageSample> samples_;
        std::vector<std::size_t> order_;

        ImageDecoder decoder_;
        ImageTransform transform_;

        Device device_;
        DataLoaderOptions options_;

        std::size_t cursor_ = 0;
    };

}

#endif // KL_DATA_LOADER_HPP