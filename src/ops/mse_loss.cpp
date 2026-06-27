#include <ops/mse_loss.hpp>

#include <ops/validation/mse_loss_validation.hpp>

#include <kernels/cpu/losses/mse_loss_cpu_float32.hpp>

#if defined(KL_ENABLE_CUDA)
#include <kernels/cuda/losses/mse_loss_cuda_float32.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <kernels/rocm/losses/mse_loss_rocm_float32.hiph>
#endif

#include <stdexcept>

namespace kl
{

    void mse_loss_unchecked(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &result,
        Reduction reduction)
    {
        mse_loss_unchecked(
            prediction,
            target,
            result,
            reduction,
            prediction.shape()[0]);
    }

    void mse_loss_unchecked(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &result,
        Reduction reduction,
        std::size_t valid_sample_count)
    {
        switch (prediction.device().type())
        {
        case DeviceType::CPU:
            mse_loss_cpu_float32(
                prediction,
                target,
                result,
                reduction,
                valid_sample_count);
            return;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            mse_loss_cuda_float32(
                prediction,
                target,
                result,
                reduction,
                valid_sample_count);
            return;
#else
            throw std::runtime_error(
                "CUDA mse_loss requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            mse_loss_rocm_float32(
                prediction,
                target,
                result,
                reduction,
                valid_sample_count);
            return;
#else
            throw std::runtime_error(
                "ROCm mse_loss requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error(
                "unknown DeviceType in mse_loss");
        }
    }

    void mse_loss(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &result,
        Reduction reduction)
    {
        mse_loss(
            prediction,
            target,
            result,
            reduction,
            prediction.shape()[0]);
    }

    void mse_loss(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &result,
        Reduction reduction,
        std::size_t valid_sample_count)
    {
        validate_mse_loss_inputs(
            prediction,
            target,
            result,
            reduction,
            valid_sample_count);

        mse_loss_unchecked(
            prediction,
            target,
            result,
            reduction,
            valid_sample_count);
    }

}