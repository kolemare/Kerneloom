#ifndef KL_TEST_CUBLAS_HANDLE_HPP
#define KL_TEST_CUBLAS_HANDLE_HPP

#ifdef KL_ENABLE_CUDA

#include <cublas_v2.h>

#include <stdexcept>

namespace kl::test
{

    class CublasHandle
    {
    public:
        CublasHandle()
        {
            if (cublasCreate(&handle_) != CUBLAS_STATUS_SUCCESS)
            {
                throw std::runtime_error("Failed to create cuBLAS handle");
            }
        }

        ~CublasHandle()
        {
            if (handle_ != nullptr)
            {
                cublasDestroy(handle_);
            }
        }

        CublasHandle(const CublasHandle &) = delete;
        CublasHandle &operator=(const CublasHandle &) = delete;

        [[nodiscard]] cublasHandle_t get() const
        {
            return handle_;
        }

    private:
        cublasHandle_t handle_ = nullptr;
    };

}

#endif // KL_ENABLE_CUDA

#endif // KL_TEST_CUBLAS_HANDLE_HPP