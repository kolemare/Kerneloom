#include <ops/categorical_cross_entropy.hpp>

#include <ops/validation/categorical_cross_entropy_validation.hpp>

#include <kernels/cpu/losses/categorical_cross_entropy_cpu_float32.hpp>

#if defined(KL_ENABLE_CUDA)
#include <kernels/cuda/losses/categorical_cross_entropy_cuda_float32.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <kernels/rocm/losses/categorical_cross_entropy_rocm_float32.hiph>
#endif

#include <stdexcept>

namespace kl
{

    void categorical_cross_entropy(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &result,
        Reduction reduction)
    {
        categorical_cross_entropy(
            prediction,
            target,
            result,
            reduction,
            prediction.shape()[0]);
    }

    void categorical_cross_entropy(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &result,
        Reduction reduction,
        std::size_t valid_sample_count)
    {
        validate_categorical_cross_entropy_inputs(
            prediction,
            target,
            result,
            reduction,
            valid_sample_count);

        switch (prediction.device().type())
        {
        case DeviceType::CPU:
            categorical_cross_entropy_cpu_float32(
                prediction,
                target,
                result,
                reduction,
                valid_sample_count);
            return;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            categorical_cross_entropy_cuda_float32(
                prediction,
                target,
                result,
                reduction,
                valid_sample_count);
            return;
#else
            throw std::runtime_error(
                "CUDA categorical_cross_entropy requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            categorical_cross_entropy_rocm_float32(
                prediction,
                target,
                result,
                reduction,
                valid_sample_count);
            return;
#else
            throw std::runtime_error(
                "ROCm categorical_cross_entropy requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error(
                "unknown DeviceType in categorical_cross_entropy");
        }
    }

}