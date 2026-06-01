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
                      std::move(targets)}))
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

    bool Batch::empty() const
    {
        return storage_ == nullptr;
    }

}