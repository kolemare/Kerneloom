#include <ops/backward_mse_loss.hpp>

#include <ops/validation/backward_mse_loss_validation.hpp>

#include <kernels/cpu/losses/backward_mse_loss_cpu_float32.hpp>

#if defined(KL_ENABLE_CUDA)
#include <kernels/cuda/losses/backward_mse_loss_cuda_float32.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <kernels/rocm/losses/backward_mse_loss_rocm_float32.hiph>
#endif

#include <stdexcept>

namespace kl
{

    void backward_mse_loss_unchecked(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &grad_prediction,
        Reduction reduction)
    {
        backward_mse_loss_unchecked(
            prediction,
            target,
            grad_prediction,
            reduction,
            prediction.shape()[0]);
    }

    void backward_mse_loss_unchecked(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &grad_prediction,
        Reduction reduction,
        std::size_t valid_sample_count)
    {
        switch (prediction.device().type())
        {
        case DeviceType::CPU:
            backward_mse_loss_cpu_float32(
                prediction,
                target,
                grad_prediction,
                reduction,
                valid_sample_count);
            return;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            backward_mse_loss_cuda_float32(
                prediction,
                target,
                grad_prediction,
                reduction,
                valid_sample_count);
            return;
#else
            throw std::runtime_error(
                "CUDA backward_mse_loss requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            backward_mse_loss_rocm_float32(
                prediction,
                target,
                grad_prediction,
                reduction,
                valid_sample_count);
            return;
#else
            throw std::runtime_error(
                "ROCm backward_mse_loss requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error(
                "unknown DeviceType in backward_mse_loss");
        }
    }

    void backward_mse_loss(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &grad_prediction,
        Reduction reduction)
    {
        backward_mse_loss(
            prediction,
            target,
            grad_prediction,
            reduction,
            prediction.shape()[0]);
    }

    void backward_mse_loss(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &grad_prediction,
        Reduction reduction,
        std::size_t valid_sample_count)
    {
        validate_backward_mse_loss_inputs(
            prediction,
            target,
            grad_prediction,
            reduction,
            valid_sample_count);

        backward_mse_loss_unchecked(
            prediction,
            target,
            grad_prediction,
            reduction,
            valid_sample_count);
    }

}