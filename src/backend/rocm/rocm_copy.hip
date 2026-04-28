#include <backend/rocm/rocm_copy.hiph>

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

namespace kl
{

    namespace
    {

        void check_hip(hipError_t error, const char *message)
        {
            if (error != hipSuccess)
            {
                throw std::runtime_error(
                    std::string(message) + ": " + hipGetErrorString(error));
            }
        }

        hipMemcpyKind copy_kind(DeviceType dst, DeviceType src)
        {
            if (dst == DeviceType::ROCM && src == DeviceType::CPU)
            {
                return hipMemcpyHostToDevice;
            }

            if (dst == DeviceType::CPU && src == DeviceType::ROCM)
            {
                return hipMemcpyDeviceToHost;
            }

            if (dst == DeviceType::ROCM && src == DeviceType::ROCM)
            {
                return hipMemcpyDeviceToDevice;
            }

            throw std::runtime_error("unsupported ROCm copy direction");
        }

    }

    void rocm_copy(Tensor &dst, const Tensor &src)
    {
        const hipMemcpyKind kind = copy_kind(
            dst.device().type(),
            src.device().type());

        check_hip(
            hipMemcpy(dst.data(), src.data(), src.nbytes(), kind),
            "hipMemcpy failed");
    }

}