#ifndef KL_TEST_CUDNN_HANDLE_HPP
#define KL_TEST_CUDNN_HANDLE_HPP

#ifdef KL_ENABLE_CUDA

#include <cudnn.h>

#include <stdexcept>

namespace kl::test
{

    class CudnnHandle
    {
    public:
        CudnnHandle()
        {
            const cudnnStatus_t status =
                cudnnCreate(&handle_);

            if (status != CUDNN_STATUS_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to create cuDNN handle");
            }
        }

        ~CudnnHandle()
        {
            if (handle_ != nullptr)
            {
                (void)cudnnDestroy(handle_);
            }
        }

        CudnnHandle(const CudnnHandle &) = delete;

        CudnnHandle &operator=(
            const CudnnHandle &) = delete;

        CudnnHandle(CudnnHandle &&) = delete;

        CudnnHandle &operator=(
            CudnnHandle &&) = delete;

        [[nodiscard]] cudnnHandle_t get() const
        {
            return handle_;
        }

    private:
        cudnnHandle_t handle_ =
            nullptr;
    };

}

#endif // KL_ENABLE_CUDA

#endif // KL_TEST_CUDNN_HANDLE_HPP