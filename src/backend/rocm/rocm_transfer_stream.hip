#include <backend/rocm/rocm_transfer_stream.hiph>

#include <hip/hip_runtime.h>

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
            DeviceType destination,
            DeviceType source)
        {
            if (destination ==
                    DeviceType::ROCM &&
                source ==
                    DeviceType::CPU)
            {
                return hipMemcpyHostToDevice;
            }

            if (destination ==
                    DeviceType::CPU &&
                source ==
                    DeviceType::ROCM)
            {
                return hipMemcpyDeviceToHost;
            }

            if (destination ==
                    DeviceType::ROCM &&
                source ==
                    DeviceType::ROCM)
            {
                return hipMemcpyDeviceToDevice;
            }

            throw std::runtime_error(
                "unsupported ROCm async copy direction");
        }

    }

    void *rocm_transfer_stream_create()
    {
        hipStream_t stream =
            nullptr;

        check_hip(
            hipStreamCreateWithFlags(
                &stream,
                hipStreamNonBlocking),
            "hipStreamCreateWithFlags failed");

        return reinterpret_cast<void *>(
            stream);
    }

    void rocm_transfer_stream_destroy(
        void *handle) noexcept
    {
        if (handle == nullptr)
        {
            return;
        }

        static_cast<void>(
            hipStreamDestroy(
                reinterpret_cast<hipStream_t>(
                    handle)));
    }

    void rocm_transfer_stream_copy_async(
        void *handle,
        Tensor &destination,
        const Tensor &source)
    {
        const hipMemcpyKind kind =
            copy_kind(
                destination.device().type(),
                source.device().type());

        check_hip(
            hipMemcpyAsync(
                destination.data(),
                source.data(),
                source.nbytes(),
                kind,
                reinterpret_cast<hipStream_t>(
                    handle)),
            "hipMemcpyAsync failed");
    }

    void rocm_transfer_stream_synchronize(
        void *handle)
    {
        check_hip(
            hipStreamSynchronize(
                reinterpret_cast<hipStream_t>(
                    handle)),
            "hipStreamSynchronize failed");
    }

}