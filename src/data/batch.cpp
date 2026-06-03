#include <data/batch.hpp>

#include <stdexcept>
#include <utility>

namespace kl
{

    namespace
    {

        void validate_batch(
            const Tensor &inputs,
            const Tensor &targets,
            std::size_t valid_sample_count)
        {
            if (inputs.rank() == 0 ||
                targets.rank() == 0)
            {
                throw std::runtime_error(
                    "Batch expects tensors with a batch dimension");
            }

            if (inputs.shape()[0] !=
                targets.shape()[0])
            {
                throw std::runtime_error(
                    "Batch expects matching input and target batch sizes");
            }

            if (valid_sample_count >
                inputs.shape()[0])
            {
                throw std::runtime_error(
                    "Batch valid sample count exceeds batch capacity");
            }
        }

    }

    Batch::Batch(
        std::shared_ptr<BatchStorage> storage)
        : storage_(
              std::move(storage))
    {
        if (storage_ == nullptr)
        {
            throw std::runtime_error(
                "Batch received null storage");
        }

        validate_batch(
            storage_->inputs,
            storage_->targets,
            storage_->valid_sample_count);
    }

    Batch::Batch(
        Tensor inputs,
        Tensor targets)
    {
        const std::size_t valid_sample_count =
            inputs.shape()[0];

        validate_batch(
            inputs,
            targets,
            valid_sample_count);

        storage_ =
            std::make_shared<BatchStorage>(
                BatchStorage{
                    std::move(inputs),
                    std::move(targets),
                    valid_sample_count});
    }

    Batch::Batch(
        Tensor inputs,
        Tensor targets,
        std::size_t valid_sample_count)
    {
        validate_batch(
            inputs,
            targets,
            valid_sample_count);

        storage_ =
            std::make_shared<BatchStorage>(
                BatchStorage{
                    std::move(inputs),
                    std::move(targets),
                    valid_sample_count});
    }

    Tensor &Batch::inputs()
    {
        if (storage_ == nullptr)
        {
            throw std::runtime_error(
                "Batch::inputs called on empty batch");
        }

        return storage_->inputs;
    }

    const Tensor &Batch::inputs() const
    {
        if (storage_ == nullptr)
        {
            throw std::runtime_error(
                "Batch::inputs called on empty batch");
        }

        return storage_->inputs;
    }

    Tensor &Batch::targets()
    {
        if (storage_ == nullptr)
        {
            throw std::runtime_error(
                "Batch::targets called on empty batch");
        }

        return storage_->targets;
    }

    const Tensor &Batch::targets() const
    {
        if (storage_ == nullptr)
        {
            throw std::runtime_error(
                "Batch::targets called on empty batch");
        }

        return storage_->targets;
    }

    std::size_t
    Batch::valid_sample_count() const
    {
        if (storage_ == nullptr)
        {
            throw std::runtime_error(
                "Batch::valid_sample_count called on empty batch");
        }

        return storage_->valid_sample_count;
    }

    bool Batch::empty() const
    {
        return storage_ == nullptr;
    }

}