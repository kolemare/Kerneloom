#include <core/tensor.hpp>

#include <core/copy.hpp>

#include <stdexcept>
#include <utility>

namespace kl
{

    Tensor::Tensor(
        Shape shape,
        DType dtype,
        Device device,
        Layout layout,
        Storage storage)
        : shape_(std::move(shape)),
          dtype_(dtype),
          device_(device),
          layout_(layout),
          storage_(storage),
          buffer_(shape_.numel() * dtype_size(dtype_), device_)
    {
    }

    const Shape &Tensor::shape() const
    {
        return shape_;
    }

    DType Tensor::dtype() const
    {
        return dtype_;
    }

    Device Tensor::device() const
    {
        return device_;
    }

    Layout Tensor::layout() const
    {
        return layout_;
    }

    Storage Tensor::storage() const
    {
        return storage_;
    }

    std::size_t Tensor::rank() const
    {
        return shape_.rank();
    }

    std::size_t Tensor::numel() const
    {
        return shape_.numel();
    }

    std::size_t Tensor::nbytes() const
    {
        return buffer_.nbytes();
    }

    void *Tensor::data()
    {
        return buffer_.data();
    }

    const void *Tensor::data() const
    {
        return buffer_.data();
    }

    void Tensor::reshape_inplace(Shape shape)
    {
        if (shape.numel() != shape_.numel())
        {
            throw std::runtime_error("Tensor::reshape_inplace requires same number of elements");
        }

        shape_ = std::move(shape);
    }

    void Tensor::set_layout(Layout layout)
    {
        layout_ = layout;
    }

    Tensor Tensor::to(Device device) const
    {
        Tensor result(
            shape_,
            dtype_,
            device,
            layout_,
            storage_);

        copy(result, *this);

        return result;
    }

}