#include <core/device.hpp>

#include <stdexcept>

namespace kl
{

    Device::Device(DeviceType type)
        : type_(type) {}

    DeviceType Device::type() const
    {
        return type_;
    }

    Device Device::cpu()
    {
        return Device(DeviceType::CPU);
    }

    Device Device::cuda()
    {
        return Device(DeviceType::CUDA);
    }

    Device Device::rocm()
    {
        return Device(DeviceType::ROCM);
    }

    std::string to_string(DeviceType type)
    {
        switch (type)
        {
        case DeviceType::CPU:
            return "CPU";
        case DeviceType::CUDA:
            return "CUDA";
        case DeviceType::ROCM:
            return "ROCM";
        default:
            throw std::runtime_error("Unknown DeviceType");
        }
    }

}