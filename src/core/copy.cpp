#include <core/copy.hpp>

#include <backend/cpu/cpu_copy.hpp>

#if defined(KL_ENABLE_CUDA)
#include <backend/cuda/cuda_copy.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <backend/rocm/rocm_copy.hiph>
#endif

#include <stdexcept>

namespace kl
{

    namespace
    {

        void validate_copy(const Tensor &dst, const Tensor &src)
        {
            if (dst.nbytes() != src.nbytes())
            {
                throw std::runtime_error("copy expects tensors with the same byte size");
            }

            if (dst.dtype() != src.dtype())
            {
                throw std::runtime_error("copy expects tensors with the same dtype");
            }

            if (dst.shape().dims() != src.shape().dims())
            {
                throw std::runtime_error("copy expects tensors with the same shape");
            }
        }

    }

    void copy(Tensor &dst, const Tensor &src)
    {
        validate_copy(dst, src);

        const DeviceType dst_type = dst.device().type();
        const DeviceType src_type = src.device().type();

        if (dst_type == DeviceType::CPU && src_type == DeviceType::CPU)
        {
            cpu_copy(dst.data(), src.data(), src.nbytes());
            return;
        }

        if (dst_type == DeviceType::CUDA || src_type == DeviceType::CUDA)
        {
#if defined(KL_ENABLE_CUDA)
            cuda_copy(dst, src);
            return;
#else
            throw std::runtime_error("CUDA copy requested but CUDA backend is not enabled");
#endif
        }

        if (dst_type == DeviceType::ROCM || src_type == DeviceType::ROCM)
        {
#if defined(KL_ENABLE_ROCM)
            rocm_copy(dst, src);
            return;
#else
            throw std::runtime_error("ROCm copy requested but ROCm backend is not enabled");
#endif
        }

        throw std::runtime_error("unsupported copy device combination");
    }

}