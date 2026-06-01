#ifndef KL_BUFFER_HPP
#define KL_BUFFER_HPP

#include <core/device.hpp>
#include <core/memory_type.hpp>

#include <cstddef>

namespace kl
{

    class Buffer
    {
    public:
        Buffer();

        Buffer(
            std::size_t nbytes,
            Device device,
            MemoryType memory_type =
                MemoryType::Default);

        ~Buffer();

        Buffer(const Buffer &) = delete;
        Buffer &operator=(const Buffer &) = delete;

        Buffer(Buffer &&other) noexcept;
        Buffer &operator=(Buffer &&other) noexcept;

        void *data();
        const void *data() const;

        std::size_t nbytes() const;

        Device device() const;
        MemoryType memory_type() const;

        bool empty() const;

    private:
        void release();

    private:
        void *data_ = nullptr;

        std::size_t nbytes_ = 0;

        Device device_ =
            Device::cpu();

        MemoryType memory_type_ =
            MemoryType::Default;
    };

}

#endif // KL_BUFFER_HPP