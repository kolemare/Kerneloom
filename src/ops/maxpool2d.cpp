#include <ops/maxpool2d.hpp>

#include <ops/validation/maxpool2d_validation.hpp>

#include <kernels/cpu/pooling/maxpool2d_cpu_float32.hpp>

#if defined(KL_ENABLE_CUDA)
#include <kernels/cuda/pooling/maxpool2d_cuda_float32.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <kernels/rocm/pooling/maxpool2d_rocm_float32.hiph>
#endif

#include <stdexcept>

namespace kl
{

    void maxpool2d(
        const Tensor &input,
        Tensor &result,
        const Pooling2dOptions &options)
    {
        validate_maxpool2d_inputs(
            input,
            result,
            options);

        switch (input.device().type())
        {
        case DeviceType::CPU:
            maxpool2d_cpu_float32(
                input,
                result,
                options);
            return;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            maxpool2d_cuda_float32(
                input,
                result,
                options);
            return;
#else
            throw std::runtime_error(
                "CUDA maxpool2d requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            maxpool2d_rocm_float32(
                input,
                result,
                options);
            return;
#else
            throw std::runtime_error(
                "ROCm maxpool2d requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error(
                "unknown DeviceType in maxpool2d");
        }
    }

}