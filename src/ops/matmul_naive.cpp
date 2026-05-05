#include <ops/matmul_naive.hpp>

#include <kernels/cpu/naive/matmul_cpu_float32_naive.hpp>

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

        void validate_matmul_inputs(
            const Tensor &a,
            const Tensor &b,
            const Tensor &c)
        {
            if (a.device().type() != b.device().type() ||
                a.device().type() != c.device().type())
            {
                throw std::runtime_error("matmul expects all tensors on the same device");
            }

            if (a.dtype() != b.dtype() ||
                a.dtype() != c.dtype())
            {
                throw std::runtime_error("matmul expects all tensors with the same dtype");
            }

            if (a.dtype() != DType::Float32)
            {
                throw std::runtime_error("matmul currently supports only Float32 tensors");
            }

            if (a.rank() != 2 || b.rank() != 2 || c.rank() != 2)
            {
                throw std::runtime_error("matmul expects rank-2 tensors");
            }

            if (a.storage() != Storage::RowMajor ||
                b.storage() != Storage::RowMajor ||
                c.storage() != Storage::RowMajor)
            {
                throw std::runtime_error("matmul currently supports only RowMajor tensors");
            }

            const std::size_t m = a.shape()[0];
            const std::size_t k_a = a.shape()[1];
            const std::size_t k_b = b.shape()[0];
            const std::size_t n = b.shape()[1];

            if (k_a != k_b)
            {
                throw std::runtime_error("matmul shape mismatch");
            }

            if (c.shape()[0] != m || c.shape()[1] != n)
            {
                throw std::runtime_error("matmul result tensor has incorrect shape");
            }
        }

    }

    void matmul_naive(
        const Tensor &a,
        const Tensor &b,
        Tensor &c)
    {
        validate_matmul_inputs(a, b, c);

        switch (a.device().type())
        {
        case DeviceType::CPU:
            matmul_cpu_float32_naive(a, b, c);
            return;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            matmul_cuda_float32_naive(a, b, c);
            return;
#else
            throw std::runtime_error("CUDA matmul requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            matmul_rocm_float32_naive(a, b, c);
            return;
#else
            throw std::runtime_error("ROCm matmul requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error("unknown DeviceType in matmul");
        }
    }

}