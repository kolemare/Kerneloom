#include <ops/backward_categorical_cross_entropy.hpp>

#include <ops/validation/backward_categorical_cross_entropy_validation.hpp>

#include <kernels/cpu/losses/backward_categorical_cross_entropy_cpu_float32.hpp>

#if defined(KL_ENABLE_CUDA)
#include <kernels/cuda/losses/backward_categorical_cross_entropy_cuda_float32.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <kernels/rocm/losses/backward_categorical_cross_entropy_rocm_float32.hiph>
#endif

#include <stdexcept>

namespace kl
{

    void backward_categorical_cross_entropy_unchecked(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &grad_prediction,
        Reduction reduction)
    {
        backward_categorical_cross_entropy_unchecked(
            prediction,
            target,
            grad_prediction,
            reduction,
            prediction.shape()[0]);
    }

    void backward_categorical_cross_entropy_unchecked(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &grad_prediction,
        Reduction reduction,
        std::size_t valid_sample_count)
    {
        switch (prediction.device().type())
        {
        case DeviceType::CPU:
            backward_categorical_cross_entropy_cpu_float32(
                prediction,
                target,
                grad_prediction,
                reduction,
                valid_sample_count);
            return;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            backward_categorical_cross_entropy_cuda_float32(
                prediction,
                target,
                grad_prediction,
                reduction,
                valid_sample_count);
            return;
#else
            throw std::runtime_error(
                "CUDA backward_categorical_cross_entropy requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            backward_categorical_cross_entropy_rocm_float32(
                prediction,
                target,
                grad_prediction,
                reduction,
                valid_sample_count);
            return;
#else
            throw std::runtime_error(
                "ROCm backward_categorical_cross_entropy requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error(
                "unknown DeviceType in backward_categorical_cross_entropy");
        }
    }

    void backward_categorical_cross_entropy(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &grad_prediction,
        Reduction reduction)
    {
        backward_categorical_cross_entropy(
            prediction,
            target,
            grad_prediction,
            reduction,
            prediction.shape()[0]);
    }

    void backward_categorical_cross_entropy(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &grad_prediction,
        Reduction reduction,
        std::size_t valid_sample_count)
    {
        validate_backward_categorical_cross_entropy_inputs(
            prediction,
            target,
            grad_prediction,
            reduction,
            valid_sample_count);

        backward_categorical_cross_entropy_unchecked(
            prediction,
            target,
            grad_prediction,
            reduction,
            valid_sample_count);
    }

}