#include <data/data_loader.hpp>

#include <core/dtype.hpp>
#include <core/layout.hpp>
#include <core/shape.hpp>
#include <core/storage.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace kl
{

    DataLoader::DataLoader(
        std::vector<ImageSample> samples,
        ImageTransform transform,
        Device device,
        DataLoaderOptions options)
        : samples_(
              std::move(samples)),
          transform_(
              std::move(transform)),
          device_(
              device),
          options_(
              options)
    {
        if (samples_.empty())
        {
            throw std::runtime_error(
                "DataLoader received an empty dataset");
        }

        if (options_.batch_size == 0)
        {
            throw std::runtime_error(
                "DataLoader batch size must be greater than zero");
        }

        reset_epoch();
    }

    void DataLoader::reset_epoch()
    {
        order_.resize(
            samples_.size());

        std::iota(
            order_.begin(),
            order_.end(),
            0);

        if (options_.shuffle)
        {
            std::mt19937 generator(
                options_.seed);

            std::shuffle(
                order_.begin(),
                order_.end(),
                generator);
        }

        cursor_ = 0;
    }

    bool DataLoader::has_next() const
    {
        if (cursor_ >= samples_.size())
        {
            return false;
        }

        const std::size_t remaining =
            samples_.size() -
            cursor_;

        if (options_.drop_last &&
            remaining <
                options_.batch_size)
        {
            return false;
        }

        return true;
    }

    Batch DataLoader::next()
    {
        if (!has_next())
        {
            throw std::runtime_error(
                "DataLoader::next called without remaining batch");
        }

        const std::size_t remaining =
            samples_.size() -
            cursor_;

        const std::size_t current_batch_size =
            std::min(
                options_.batch_size,
                remaining);

        const std::size_t channels =
            transform_.output_channels();

        const std::size_t height =
            transform_.output_height();

        const std::size_t width =
            transform_.output_width();

        Tensor inputs_cpu(
            Shape{
                current_batch_size,
                channels,
                height,
                width},
            DType::Float32,
            Device::cpu(),
            Layout::NCHW,
            Storage::RowMajor);

        Tensor targets_cpu(
            Shape{
                current_batch_size},
            DType::Int32,
            Device::cpu(),
            Layout::Unknown,
            Storage::RowMajor);

        float *inputs_data =
            static_cast<float *>(
                inputs_cpu.data());

        std::int32_t *targets_data =
            static_cast<std::int32_t *>(
                targets_cpu.data());

        const std::size_t image_numel =
            transform_.output_numel();

        for (std::size_t n = 0;
             n < current_batch_size;
             ++n)
        {
            const ImageSample &sample =
                samples_[order_[cursor_ +
                                n]];

            const Image image =
                decoder_.decode(
                    sample.path);

            transform_.write_chw(
                image,
                inputs_data +
                    n *
                        image_numel);

            targets_data[n] =
                sample.label;
        }

        cursor_ +=
            current_batch_size;

        if (device_.type() ==
            DeviceType::CPU)
        {
            return Batch{
                std::move(inputs_cpu),
                std::move(targets_cpu)};
        }

        return Batch{
            inputs_cpu.to(device_),
            targets_cpu.to(device_)};
    }

    std::size_t DataLoader::sample_count() const
    {
        return samples_.size();
    }

    std::size_t DataLoader::batch_count() const
    {
        if (options_.drop_last)
        {
            return samples_.size() /
                   options_.batch_size;
        }

        return (
                   samples_.size() +
                   options_.batch_size -
                   1) /
               options_.batch_size;
    }

}