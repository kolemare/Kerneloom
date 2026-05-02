#include <ops/matmul.hpp>

#include <kernels/cpu/matmul_cpu_float32_naive.hpp>

#if defined(KL_ENABLE_CUDA)
#include <kernels/cuda/matmul_cuda_float32_naive.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <kernels/rocm/matmul_rocm_float32_naive.hiph>
#endif

#include <stdexcept>

namespace kl
{

    namespace
    {

        void validate_matmul_inputs(const Tensor &a, const Tensor &b)
        {
            if (a.device().type() != b.device().type())
            {
                throw std::runtime_error("matmul expects both tensors on the same device");
            }

            if (a.dtype() != b.dtype())
            {
                throw std::runtime_error("matmul expects tensors with the same dtype");
            }

            if (a.dtype() != DType::Float32)
            {
                throw std::runtime_error("matmul currently supports only Float32 tensors");
            }

            if (a.rank() != 2 || b.rank() != 2)
            {
                throw std::runtime_error("matmul expects two rank-2 tensors");
            }

            if (a.storage() != Storage::RowMajor ||
                b.storage() != Storage::RowMajor)
            {
                throw std::runtime_error("matmul currently supports only RowMajor tensors");
            }

            const std::size_t a_cols = a.shape()[1];
            const std::size_t b_rows = b.shape()[0];

            if (a_cols != b_rows)
            {
                throw std::runtime_error("matmul shape mismatch");
            }
        }

    }

    Tensor matmul(const Tensor &a, const Tensor &b)
    {
        validate_matmul_inputs(a, b);

        const std::size_t m = a.shape()[0];
        const std::size_t n = b.shape()[1];

        Tensor c(
            Shape{m, n},
            a.dtype(),
            a.device(),
            Layout::Unknown,
            Storage::RowMajor);

        switch (a.device().type())
        {
        case DeviceType::CPU:
            matmul_cpu_float32_naive(a, b, c);
            return c;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            matmul_cuda_float32_naive(a, b, c);
            return c;
#else
            throw std::runtime_error("CUDA matmul requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            matmul_rocm_float32_naive(a, b, c);
            return c;
#else
            throw std::runtime_error("ROCm matmul requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error("unknown DeviceType in matmul");
        }
    }

}