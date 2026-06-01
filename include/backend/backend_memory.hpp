#ifndef KL_BACKEND_MEMORY_HPP
#define KL_BACKEND_MEMORY_HPP

#include <core/device.hpp>

#include <cstddef>

namespace kl
{

    std::size_t available_device_memory_bytes(
        Device device);

}

#endif // KL_BACKEND_MEMORY_HPP