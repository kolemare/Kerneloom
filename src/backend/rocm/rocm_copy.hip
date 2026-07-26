#include <backend/rocm/rocm_copy.hiph>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace kl
{

    namespace
    {

        void check_hip(
            hipError_t error,
            const char *message)
        {
            if (error != hipSuccess)
            {
                throw std::runtime_error(
                    std::string(message) +
                    ": " +
                    hipGetErrorString(error));
            }
        }

        hipMemcpyKind copy_kind(
            DeviceType dst,
            DeviceType src)
        {
            if (dst == DeviceType::ROCM &&
                src == DeviceType::CPU)
            {
                return hipMemcpyHostToDevice;
            }

            if (dst == DeviceType::CPU &&
                src == DeviceType::ROCM)
            {
                return hipMemcpyDeviceToHost;
            }

            if (dst == DeviceType::ROCM &&
                src == DeviceType::ROCM)
            {
                return hipMemcpyDeviceToDevice;
            }

            throw std::runtime_error(
                "unsupported ROCm copy direction");
        }

    } // namespace

    void rocm_copy(
        Tensor &dst,
        const Tensor &src)
    {
        const hipMemcpyKind kind =
            copy_kind(
                dst.device().type(),
                src.device().type());

        check_hip(
            hipMemcpy(
                dst.data(),
                src.data(),
                src.nbytes(),
                kind),
            "hipMemcpy failed");
    }

    void rocm_copy_to_host(
        void *destination,
        const Tensor &source,
        std::size_t source_offset_bytes,
        std::size_t nbytes)
    {
        if (source.device().type() !=
            DeviceType::ROCM)
        {
            throw std::runtime_error(
                "rocm_copy_to_host requires a ROCm source tensor");
        }

        if (nbytes == 0)
        {
            return;
        }

        const auto *source_bytes =
            static_cast<const std::byte *>(
                source.data());

        check_hip(
            hipMemcpy(
                destination,
                source_bytes + source_offset_bytes,
                nbytes,
                hipMemcpyDeviceToHost),
            "ROCm chunk readback failed");
    }

} // namespace kl