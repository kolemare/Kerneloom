#include <ops/conv2d.hpp>

#include <ops/validation/conv2d_validation.hpp>

#include <kernels/cpu/conv2d/conv2d_cpu_float32_dispatcher.hpp>

#if defined(KL_ENABLE_CUDA)
#include <kernels/cuda/conv2d/conv2d_cuda_float32_dispatcher.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <kernels/rocm/conv2d/conv2d_rocm_float32_dispatcher.hiph>
#endif

#include <stdexcept>

namespace kl
{

    void conv2d(
        const Tensor &input,
        const Tensor &kernels,
        const Tensor *bias,
        Tensor &result,
        const Conv2dOptions &options)
    {
        validate_conv2d_inputs(
            input,
            kernels,
            bias,
            result,
            options);

        switch (input.device().type())
        {
        case DeviceType::CPU:
            conv2d_cpu_float32_dispatcher(
                input,
                kernels,
                bias,
                result,
                options);
            return;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            conv2d_cuda_float32_dispatcher(
                input,
                kernels,
                bias,
                result,
                options);
            return;
#else
            throw std::runtime_error(
                "CUDA conv2d requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            conv2d_rocm_float32_dispatcher(
                input,
                kernels,
                bias,
                result,
                options);
            return;
#else
            throw std::runtime_error(
                "ROCm conv2d requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error(
                "unknown DeviceType in conv2d");
        }
    }

}