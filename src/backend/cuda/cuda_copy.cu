#include <backend/cuda/cuda_copy.cuh>

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace kl
{

    namespace
    {

        void check_cuda(
            cudaError_t error,
            const char *message)
        {
            if (error != cudaSuccess)
            {
                throw std::runtime_error(
                    std::string(message) +
                    ": " +
                    cudaGetErrorString(error));
            }
        }

        cudaMemcpyKind copy_kind(
            DeviceType dst,
            DeviceType src)
        {
            if (dst == DeviceType::CUDA &&
                src == DeviceType::CPU)
            {
                return cudaMemcpyHostToDevice;
            }

            if (dst == DeviceType::CPU &&
                src == DeviceType::CUDA)
            {
                return cudaMemcpyDeviceToHost;
            }

            if (dst == DeviceType::CUDA &&
                src == DeviceType::CUDA)
            {
                return cudaMemcpyDeviceToDevice;
            }

            throw std::runtime_error(
                "unsupported CUDA copy direction");
        }

    } // namespace

    void cuda_copy(
        Tensor &dst,
        const Tensor &src)
    {
        const cudaMemcpyKind kind =
            copy_kind(
                dst.device().type(),
                src.device().type());

        check_cuda(
            cudaMemcpy(
                dst.data(),
                src.data(),
                src.nbytes(),
                kind),
            "cudaMemcpy failed");
    }

    void cuda_copy_to_host(
        void *destination,
        const Tensor &source,
        std::size_t source_offset_bytes,
        std::size_t nbytes)
    {
        if (source.device().type() !=
            DeviceType::CUDA)
        {
            throw std::runtime_error(
                "cuda_copy_to_host requires a CUDA source tensor");
        }

        if (nbytes == 0)
        {
            return;
        }

        const auto *source_bytes =
            static_cast<const std::byte *>(
                source.data());

        check_cuda(
            cudaMemcpy(
                destination,
                source_bytes + source_offset_bytes,
                nbytes,
                cudaMemcpyDeviceToHost),
            "CUDA chunk readback failed");
    }

} // namespace kl