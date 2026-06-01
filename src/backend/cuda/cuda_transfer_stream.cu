#include <backend/cuda/cuda_transfer_stream.cuh>

#include <cuda_runtime.h>

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
            DeviceType destination,
            DeviceType source)
        {
            if (destination ==
                    DeviceType::CUDA &&
                source ==
                    DeviceType::CPU)
            {
                return cudaMemcpyHostToDevice;
            }

            if (destination ==
                    DeviceType::CPU &&
                source ==
                    DeviceType::CUDA)
            {
                return cudaMemcpyDeviceToHost;
            }

            if (destination ==
                    DeviceType::CUDA &&
                source ==
                    DeviceType::CUDA)
            {
                return cudaMemcpyDeviceToDevice;
            }

            throw std::runtime_error(
                "unsupported CUDA async copy direction");
        }

    }

    void *cuda_transfer_stream_create()
    {
        cudaStream_t stream =
            nullptr;

        check_cuda(
            cudaStreamCreateWithFlags(
                &stream,
                cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags failed");

        return reinterpret_cast<void *>(
            stream);
    }

    void cuda_transfer_stream_destroy(
        void *handle) noexcept
    {
        if (handle == nullptr)
        {
            return;
        }

        static_cast<void>(
            cudaStreamDestroy(
                reinterpret_cast<cudaStream_t>(
                    handle)));
    }

    void cuda_transfer_stream_copy_async(
        void *handle,
        Tensor &destination,
        const Tensor &source)
    {
        const cudaMemcpyKind kind =
            copy_kind(
                destination.device().type(),
                source.device().type());

        check_cuda(
            cudaMemcpyAsync(
                destination.data(),
                source.data(),
                source.nbytes(),
                kind,
                reinterpret_cast<cudaStream_t>(
                    handle)),
            "cudaMemcpyAsync failed");
    }

    void cuda_transfer_stream_synchronize(
        void *handle)
    {
        check_cuda(
            cudaStreamSynchronize(
                reinterpret_cast<cudaStream_t>(
                    handle)),
            "cudaStreamSynchronize failed");
    }

}