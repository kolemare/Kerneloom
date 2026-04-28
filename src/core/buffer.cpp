#include <core/buffer.hpp>

#include <core/allocator.hpp>

#include <utility>

namespace kl
{

    Buffer::Buffer() = default;

    Buffer::Buffer(std::size_t nbytes, Device device)
        : nbytes_(nbytes),
          device_(device)
    {
        data_ = allocator_for(device_).allocate(nbytes_);
    }

    Buffer::~Buffer()
    {
        release();
    }

    Buffer::Buffer(Buffer &&other) noexcept
        : data_(other.data_),
          nbytes_(other.nbytes_),
          device_(other.device_)
    {
        other.data_ = nullptr;
        other.nbytes_ = 0;
        other.device_ = Device::cpu();
    }

    Buffer &Buffer::operator=(Buffer &&other) noexcept
    {
        if (this != &other)
        {
            release();

            data_ = other.data_;
            nbytes_ = other.nbytes_;
            device_ = other.device_;

            other.data_ = nullptr;
            other.nbytes_ = 0;
            other.device_ = Device::cpu();
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

    bool Buffer::empty() const
    {
        return data_ == nullptr || nbytes_ == 0;
    }

    void Buffer::release()
    {
        if (data_ != nullptr)
        {
            allocator_for(device_).deallocate(data_);
            data_ = nullptr;
            nbytes_ = 0;
            device_ = Device::cpu();
        }
    }

}