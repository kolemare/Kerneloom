#include <core/copy.hpp>

#include <core/synchronize.hpp>

#include <backend/cpu/cpu_copy.hpp>

#if defined(KL_ENABLE_CUDA)
#include <backend/cuda/cuda_copy.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <backend/rocm/rocm_copy.hiph>
#endif

#include <cstddef>
#include <stdexcept>

namespace kl
{

    namespace
    {

        void validate_copy(
            const Tensor &dst,
            const Tensor &src)
        {
            if (dst.nbytes() != src.nbytes())
            {
                throw std::runtime_error(
                    "copy expects tensors with the same byte size");
            }

            if (dst.dtype() != src.dtype())
            {
                throw std::runtime_error(
                    "copy expects tensors with the same dtype");
            }

            if (dst.shape().dims() != src.shape().dims())
            {
                throw std::runtime_error(
                    "copy expects tensors with the same shape");
            }
        }

        void validate_copy_to_host(
            void *destination,
            const Tensor &source,
            std::size_t source_offset_bytes,
            std::size_t nbytes)
        {
            if (nbytes == 0)
            {
                return;
            }

            if (destination == nullptr)
            {
                throw std::runtime_error(
                    "copy_to_host destination cannot be null");
            }

            if (source_offset_bytes > source.nbytes())
            {
                throw std::runtime_error(
                    "copy_to_host source offset is out of range");
            }

            const std::size_t available_bytes =
                source.nbytes() - source_offset_bytes;

            if (nbytes > available_bytes)
            {
                throw std::runtime_error(
                    "copy_to_host range exceeds source tensor size");
            }
        }

        bool is_host_readback(
            DeviceType dst_type,
            DeviceType src_type)
        {
            return dst_type == DeviceType::CPU &&
                   src_type != DeviceType::CPU;
        }

    } // namespace

    void copy(
        Tensor &dst,
        const Tensor &src)
    {
        validate_copy(
            dst,
            src);

        const DeviceType dst_type =
            dst.device().type();

        const DeviceType src_type =
            src.device().type();

        if (dst_type == DeviceType::CPU &&
            src_type == DeviceType::CPU)
        {
            cpu_copy(
                dst.data(),
                src.data(),
                src.nbytes());

            return;
        }

        if (dst_type == DeviceType::CUDA ||
            src_type == DeviceType::CUDA)
        {
#if defined(KL_ENABLE_CUDA)
            cuda_copy(
                dst,
                src);

            if (is_host_readback(
                    dst_type,
                    src_type))
            {
                synchronize(
                    src.device());
            }

            return;
#else
            throw std::runtime_error(
                "CUDA copy requested but CUDA backend is not enabled");
#endif
        }

        if (dst_type == DeviceType::ROCM ||
            src_type == DeviceType::ROCM)
        {
#if defined(KL_ENABLE_ROCM)
            rocm_copy(
                dst,
                src);

            if (is_host_readback(
                    dst_type,
                    src_type))
            {
                synchronize(
                    src.device());
            }

            return;
#else
            throw std::runtime_error(
                "ROCm copy requested but ROCm backend is not enabled");
#endif
        }

        throw std::runtime_error(
            "unsupported copy device combination");
    }

    void copy_to_host(
        void *destination,
        const Tensor &source,
        std::size_t source_offset_bytes,
        std::size_t nbytes)
    {
        validate_copy_to_host(
            destination,
            source,
            source_offset_bytes,
            nbytes);

        if (nbytes == 0)
        {
            return;
        }

        switch (source.device().type())
        {
        case DeviceType::CPU:
        {
            const auto *source_bytes =
                static_cast<const std::byte *>(
                    source.data());

            cpu_copy(
                destination,
                source_bytes + source_offset_bytes,
                nbytes);

            return;
        }

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            cuda_copy_to_host(
                destination,
                source,
                source_offset_bytes,
                nbytes);

            return;
#else
            throw std::runtime_error(
                "CUDA readback requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            rocm_copy_to_host(
                destination,
                source,
                source_offset_bytes,
                nbytes);

            return;
#else
            throw std::runtime_error(
                "ROCm readback requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error(
                "unsupported source device in copy_to_host");
        }
    }

} // namespace kl