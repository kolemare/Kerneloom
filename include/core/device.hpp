#ifndef KL_DEVICE_HPP
#define KL_DEVICE_HPP

#include <string>

namespace kl
{

    enum class DeviceType
    {
        CPU,
        CUDA,
        ROCM
    };

    class Device
    {
    public:
        explicit Device(DeviceType type);

        DeviceType type() const;

        static Device cpu();
        static Device cuda();
        static Device rocm();

    private:
        DeviceType type_;
    };

    std::string to_string(DeviceType type);

}

#endif // KL_DEVICE_HPP
