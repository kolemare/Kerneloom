#include <core/tensor_pool.hpp>

namespace kl
{

    Tensor &TensorPool::request(
        const Shape &shape,
        DType dtype,
        Device device,
        Layout layout,
        Storage storage)
    {
        TensorEntry *entry = find_reusable(
            shape,
            dtype,
            device,
            layout,
            storage);

        if (entry != nullptr)
        {
            entry->tensor->reshape_inplace(shape);
            entry->tensor->set_layout(layout);
            entry->active = true;
            return *entry->tensor;
        }

        TensorEntry new_entry;
        new_entry.tensor = std::make_unique<Tensor>(
            shape,
            dtype,
            device,
            layout,
            storage);
        new_entry.active = true;

        tensors_.push_back(std::move(new_entry));

        return *tensors_.back().tensor;
    }

    void TensorPool::reset()
    {
        for (TensorEntry &entry : tensors_)
        {
            entry.active = false;
        }
    }

    void TensorPool::clear()
    {
        tensors_.clear();
    }

    std::size_t TensorPool::tensor_count() const
    {
        return tensors_.size();
    }

    std::size_t TensorPool::active_count() const
    {
        std::size_t count = 0;

        for (const TensorEntry &entry : tensors_)
        {
            if (entry.active)
            {
                ++count;
            }
        }

        return count;
    }

    std::size_t TensorPool::inactive_count() const
    {
        return tensor_count() - active_count();
    }

    TensorPool::TensorEntry *TensorPool::find_reusable(
        const Shape &shape,
        DType dtype,
        Device device,
        Layout layout,
        Storage storage)
    {
        for (TensorEntry &entry : tensors_)
        {
            if (entry.active)
            {
                continue;
            }

            if (matches(
                    *entry.tensor,
                    shape,
                    dtype,
                    device,
                    layout,
                    storage))
            {
                return &entry;
            }
        }

        return nullptr;
    }

    bool TensorPool::matches(
        const Tensor &tensor,
        const Shape &shape,
        DType dtype,
        Device device,
        Layout layout,
        Storage storage) const
    {
        (void)layout;

        return tensor.dtype() == dtype &&
               tensor.device().type() == device.type() &&
               tensor.storage() == storage &&
               tensor.nbytes() == shape.numel() * dtype_size(dtype);
    }

}