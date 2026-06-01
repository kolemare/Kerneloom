#include <core/transfer_stream.hpp>

#include <core/copy.hpp>

#if defined(KL_ENABLE_CUDA)
#include <backend/cuda/cuda_transfer_stream.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <backend/rocm/rocm_transfer_stream.hiph>
#endif

#include <stdexcept>

namespace kl
{

    TransferStream::TransferStream(
        Device device)
        : device_(
              device)
    {
        switch (device_.type())
        {
        case DeviceType::CPU:
            return;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            handle_ =
                cuda_transfer_stream_create();
            return;
#else
            throw std::runtime_error(
                "CUDA transfer stream requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            handle_ =
                rocm_transfer_stream_create();
            return;
#else
            throw std::runtime_error(
                "ROCm transfer stream requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error(
                "unknown DeviceType in TransferStream");
        }
    }

    TransferStream::~TransferStream()
    {
        switch (device_.type())
        {
        case DeviceType::CPU:
            return;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            cuda_transfer_stream_destroy(
                handle_);
#endif
            return;

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            rocm_transfer_stream_destroy(
                handle_);
#endif
            return;

        default:
            return;
        }
    }

    void TransferStream::copy_async(
        Tensor &destination,
        const Tensor &source)
    {
        if (destination.nbytes() !=
            source.nbytes())
        {
            throw std::runtime_error(
                "TransferStream::copy_async expects equal byte sizes");
        }

        if (destination.dtype() !=
            source.dtype())
        {
            throw std::runtime_error(
                "TransferStream::copy_async expects equal dtypes");
        }

        if (destination.shape() !=
            source.shape())
        {
            throw std::runtime_error(
                "TransferStream::copy_async expects equal shapes");
        }

        switch (device_.type())
        {
        case DeviceType::CPU:
            copy(
                destination,
                source);
            return;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            cuda_transfer_stream_copy_async(
                handle_,
                destination,
                source);
            return;
#else
            throw std::runtime_error(
                "CUDA transfer requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            rocm_transfer_stream_copy_async(
                handle_,
                destination,
                source);
            return;
#else
            throw std::runtime_error(
                "ROCm transfer requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error(
                "unknown DeviceType in TransferStream::copy_async");
        }
    }

    void TransferStream::synchronize()
    {
        switch (device_.type())
        {
        case DeviceType::CPU:
            return;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            cuda_transfer_stream_synchronize(
                handle_);
            return;
#else
            throw std::runtime_error(
                "CUDA transfer stream requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            rocm_transfer_stream_synchronize(
                handle_);
            return;
#else
            throw std::runtime_error(
                "ROCm transfer stream requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error(
                "unknown DeviceType in TransferStream::synchronize");
        }
    }

    Device TransferStream::device() const
    {
        return device_;
    }

}