#include <core/buffer.hpp>

#include <core/allocator.hpp>

namespace kl
{

    Buffer::Buffer() = default;

    Buffer::Buffer(
        std::size_t nbytes,
        Device device,
        MemoryType memory_type)
        : nbytes_(nbytes),
          device_(device),
          memory_type_(memory_type)
    {
        data_ =
            allocator_for(
                device_,
                memory_type_)
                .allocate(
                    nbytes_);
    }

    Buffer::~Buffer() noexcept
    {
        release();
    }

    Buffer::Buffer(
        Buffer &&other) noexcept
        : data_(other.data_),
          nbytes_(other.nbytes_),
          device_(other.device_),
          memory_type_(other.memory_type_)
    {
        other.data_ = nullptr;
        other.nbytes_ = 0;

        other.device_ =
            Device::cpu();

        other.memory_type_ =
            MemoryType::Default;
    }

    Buffer &Buffer::operator=(
        Buffer &&other) noexcept
    {
        if (this != &other)
        {
            release();

            data_ =
                other.data_;

            nbytes_ =
                other.nbytes_;

            device_ =
                other.device_;

            memory_type_ =
                other.memory_type_;

            other.data_ = nullptr;
            other.nbytes_ = 0;

            other.device_ =
                Device::cpu();

            other.memory_type_ =
                MemoryType::Default;
        }

        return *this;
    }

    void *Buffer::data()
    {
        return data_;
    }

    const void *Buffer::data() const
    {
        return data_;
    }

    std::size_t Buffer::nbytes() const
    {
        return nbytes_;
    }

    Device Buffer::device() const
    {
        return device_;
    }

    MemoryType Buffer::memory_type() const
    {
        return memory_type_;
    }

    bool Buffer::empty() const
    {
        return data_ == nullptr ||
               nbytes_ == 0;
    }

    void Buffer::release() noexcept
    {
        if (data_ == nullptr)
        {
            return;
        }

        allocator_for(
            device_,
            memory_type_)
            .deallocate(
                data_);

        data_ = nullptr;
        nbytes_ = 0;

        device_ =
            Device::cpu();

        memory_type_ =
            MemoryType::Default;
    }

}