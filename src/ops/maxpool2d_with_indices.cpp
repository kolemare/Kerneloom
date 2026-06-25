#include <ops/maxpool2d_with_indices.hpp>

#include <ops/validation/maxpool2d_with_indices_validation.hpp>

#include <kernels/cpu/pooling/maxpool2d_with_indices_cpu_float32.hpp>

#if defined(KL_ENABLE_CUDA)
#include <kernels/cuda/pooling/maxpool2d_with_indices_cuda_float32.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <kernels/rocm/pooling/maxpool2d_with_indices_rocm_float32.hiph>
#endif

#include <stdexcept>

namespace kl
{

    void maxpool2d_with_indices(
        const Tensor &input,
        Tensor &result,
        Tensor &indices,
        const Pooling2dOptions &options)
    {
        validate_maxpool2d_with_indices_inputs(
            input,
            result,
            indices,
            options);

        switch (input.device().type())
        {
        case DeviceType::CPU:
            maxpool2d_with_indices_cpu_float32(
                input,
                result,
                indices,
                options);
            return;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            maxpool2d_with_indices_cuda_float32(
                input,
                result,
                indices,
                options);
            return;
#else
            throw std::runtime_error(
                "CUDA maxpool2d_with_indices requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            maxpool2d_with_indices_rocm_float32(
                input,
                result,
                indices,
                options);
            return;
#else
            throw std::runtime_error(
                "ROCm maxpool2d_with_indices requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error(
                "unknown DeviceType in maxpool2d_with_indices");
        }
    }

}