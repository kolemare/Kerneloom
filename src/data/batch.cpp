#include <data/batch.hpp>

#include <stdexcept>
#include <utility>

namespace kl
{

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
    }

    Batch::Batch(
        Tensor inputs,
        Tensor targets)
        : storage_(
              std::make_shared<BatchStorage>(
                  BatchStorage{
                      std::move(inputs),
                      std::move(targets),
                      inputs.shape()[0]}))
    {
    }

    Batch::Batch(
        Tensor inputs,
        Tensor targets,
        std::size_t valid_sample_count)
        : storage_(
              std::make_shared<BatchStorage>(
                  BatchStorage{
                      std::move(inputs),
                      std::move(targets),
                      valid_sample_count}))
    {
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