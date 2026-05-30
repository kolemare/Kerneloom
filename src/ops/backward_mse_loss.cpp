#include <ops/backward_mse_loss.hpp>

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

    namespace
    {

        void validate_backward_mse_loss_inputs(
            const Tensor &prediction,
            const Tensor &target,
            const Tensor &grad_prediction,
            Reduction reduction)
        {
            if (prediction.device().type() != target.device().type() ||
                prediction.device().type() != grad_prediction.device().type())
            {
                throw std::runtime_error(
                    "backward_mse_loss expects tensors on the same device");
            }

            if (prediction.dtype() != target.dtype() ||
                prediction.dtype() != grad_prediction.dtype())
            {
                throw std::runtime_error(
                    "backward_mse_loss expects tensors with the same dtype");
            }

            if (prediction.dtype() != DType::Float32)
            {
                throw std::runtime_error(
                    "backward_mse_loss currently supports only Float32 tensors");
            }

            if (prediction.shape() != target.shape() ||
                prediction.shape() != grad_prediction.shape())
            {
                throw std::runtime_error(
                    "backward_mse_loss expects tensors with the same shape");
            }

            if (prediction.numel() == 0)
            {
                throw std::runtime_error(
                    "backward_mse_loss expects non-empty tensors");
            }

            if (prediction.storage() != Storage::RowMajor ||
                target.storage() != Storage::RowMajor ||
                grad_prediction.storage() != Storage::RowMajor)
            {
                throw std::runtime_error(
                    "backward_mse_loss currently supports only RowMajor tensors");
            }

            switch (reduction)
            {
            case Reduction::Mean:
            case Reduction::Sum:
                return;

            default:
                throw std::runtime_error(
                    "unknown Reduction in backward_mse_loss");
            }
        }

    }

    void backward_mse_loss(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &grad_prediction,
        Reduction reduction)
    {
        validate_backward_mse_loss_inputs(
            prediction,
            target,
            grad_prediction,
            reduction);

        switch (prediction.device().type())
        {
        case DeviceType::CPU:
            backward_mse_loss_cpu_float32(
                prediction,
                target,
                grad_prediction,
                reduction);
            return;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            backward_mse_loss_cuda_float32(
                prediction,
                target,
                grad_prediction,
                reduction);
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
                reduction);
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

}