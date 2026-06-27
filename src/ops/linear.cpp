#include <ops/linear.hpp>

#include <ops/validation/linear_validation.hpp>

#include <kernels/cpu/linear/linear_cpu_float32.hpp>

#if defined(KL_ENABLE_CUDA)
#include <kernels/cuda/linear/linear_cuda_float32.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <kernels/rocm/linear/linear_rocm_float32.hiph>
#endif

#include <stdexcept>

namespace kl
{

    void linear_unchecked(
        const Tensor &input,
        const Tensor &weights,
        const Tensor *bias,
        Tensor &result)
    {
        switch (input.device().type())
        {
        case DeviceType::CPU:
            linear_cpu_float32(
                input,
                weights,
                bias,
                result);
            return;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            linear_cuda_float32(
                input,
                weights,
                bias,
                result);
            return;
#else
            throw std::runtime_error(
                "CUDA linear requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            linear_rocm_float32(
                input,
                weights,
                bias,
                result);
            return;
#else
            throw std::runtime_error(
                "ROCm linear requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error(
                "unknown DeviceType in linear");
        }
    }

    void linear(
        const Tensor &input,
        const Tensor &weights,
        const Tensor *bias,
        Tensor &result)
    {
        validate_linear_inputs(
            input,
            weights,
            bias,
            result);

        linear_unchecked(
            input,
            weights,
            bias,
            result);
    }

}