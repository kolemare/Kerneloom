#ifndef KL_TENSOR_HPP
#define KL_TENSOR_HPP

#include <core/buffer.hpp>
#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/layout.hpp>
#include <core/memory_type.hpp>
#include <core/shape.hpp>
#include <core/storage.hpp>

#include <cstddef>

namespace kl
{

    class Tensor
    {
    public:
        Tensor(
            Shape shape,
            DType dtype = DType::Float32,
            Device device = Device::cpu(),
            Layout layout = Layout::Unknown,
            Storage storage = Storage::RowMajor,
            MemoryType memory_type =
                MemoryType::Default);

        Tensor(const Tensor &) = delete;
        Tensor &operator=(const Tensor &) = delete;

        Tensor(Tensor &&other) noexcept = default;
        Tensor &operator=(Tensor &&other) noexcept = default;

        const Shape &shape() const;

        DType dtype() const;
        Device device() const;

        Layout layout() const;
        Storage storage() const;

        MemoryType memory_type() const;

        std::size_t rank() const;
        std::size_t numel() const;
        std::size_t nbytes() const;

        void *data();
        const void *data() const;

        void reshape_inplace(
            Shape shape);

        void set_layout(
            Layout layout);

        Tensor to(
            Device device,
            MemoryType memory_type =
                MemoryType::Default) const;

    private:
        Shape shape_;

        DType dtype_;
        Device device_;

        Layout layout_;
        Storage storage_;

        Buffer buffer_;
    };

}

#endif // KL_TENSOR_HPP