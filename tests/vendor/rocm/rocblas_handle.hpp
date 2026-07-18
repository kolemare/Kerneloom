#ifndef KL_TEST_ROCBLAS_HANDLE_HPP
#define KL_TEST_ROCBLAS_HANDLE_HPP

#ifdef KL_ENABLE_ROCM

#include <rocblas/rocblas.h>

#include <stdexcept>

namespace kl::test
{

    class RocblasHandle
    {
    public:
        RocblasHandle()
        {
            if (rocblas_create_handle(&handle_) != rocblas_status_success)
            {
                throw std::runtime_error("Failed to create rocBLAS handle");
            }
        }

        ~RocblasHandle()
        {
            if (handle_ != nullptr)
            {
                rocblas_destroy_handle(handle_);
            }
        }

        RocblasHandle(const RocblasHandle &) = delete;
        RocblasHandle &operator=(const RocblasHandle &) = delete;

        [[nodiscard]] rocblas_handle get() const
        {
            return handle_;
        }

    private:
        rocblas_handle handle_ = nullptr;
    };

}

#endif // KL_ENABLE_ROCM

#endif // KL_TEST_ROCBLAS_HANDLE_HPP