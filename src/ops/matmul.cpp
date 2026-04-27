#include <ops/matmul.hpp>

#include <stdexcept>

namespace kl::detail
{

#if KERNELOOM_HAS_CPU
    void matmul_cpu(
        const float *a,
        const float *b,
        float *c,
        int m,
        int n,
        int k);
#endif

#if KERNELOOM_HAS_CUDA
    void matmul_cuda(
        const float *a,
        const float *b,
        float *c,
        int m,
        int n,
        int k);
#endif

#if KERNELOOM_HAS_ROCM
    void matmul_rocm(
        const float *a,
        const float *b,
        float *c,
        int m,
        int n,
        int k);
#endif

}

namespace kl
{

    Tensor matmul(const Tensor &a, const Tensor &b)
    {
        if (a.cols() != b.rows())
        {
            throw std::runtime_error("matmul shape mismatch");
        }

        if (a.device().type() != b.device().type())
        {
            throw std::runtime_error("matmul requires both tensors to use the same device");
        }

        const int m = static_cast<int>(a.rows());
        const int n = static_cast<int>(b.cols());
        const int k = static_cast<int>(a.cols());

        Tensor out(Shape2D(a.rows(), b.cols()), a.device());

        switch (a.device().type())
        {
        case DeviceType::CPU:
#if KERNELOOM_HAS_CPU
            detail::matmul_cpu(a.data(), b.data(), out.data(), m, n, k);
            break;
#else
            throw std::runtime_error("CPU backend was not compiled");
#endif

        case DeviceType::CUDA:
#if KERNELOOM_HAS_CUDA
            detail::matmul_cuda(a.data(), b.data(), out.data(), m, n, k);
            break;
#else
            throw std::runtime_error("CUDA backend was not compiled");
#endif

        case DeviceType::ROCM:
#if KERNELOOM_HAS_ROCM
            detail::matmul_rocm(a.data(), b.data(), out.data(), m, n, k);
            break;
#else
            throw std::runtime_error("ROCm backend was not compiled");
#endif
        }

        return out;
    }

}