#include <backend/cuda/cuda_copy.cuh>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace kl
{

    namespace
    {

        void check_cuda(cudaError_t error, const char *message)
        {
            if (error != cudaSuccess)
            {
                throw std::runtime_error(
                    std::string(message) + ": " + cudaGetErrorString(error));
            }
        }

        cudaMemcpyKind copy_kind(DeviceType dst, DeviceType src)
        {
            if (dst == DeviceType::CUDA && src == DeviceType::CPU)
            {
                return cudaMemcpyHostToDevice;
            }

            if (dst == DeviceType::CPU && src == DeviceType::CUDA)
            {
                return cudaMemcpyDeviceToHost;
            }

            if (dst == DeviceType::CUDA && src == DeviceType::CUDA)
            {
                return cudaMemcpyDeviceToDevice;
            }

            throw std::runtime_error("unsupported CUDA copy direction");
        }

    }

    void cuda_copy(Tensor &dst, const Tensor &src)
    {
        const cudaMemcpyKind kind = copy_kind(
            dst.device().type(),
            src.device().type());

        check_cuda(
            cudaMemcpy(dst.data(), src.data(), src.nbytes(), kind),
            "cudaMemcpy failed");
    }

}