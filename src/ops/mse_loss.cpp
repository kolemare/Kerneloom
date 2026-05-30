#include <ops/mse_loss.hpp>

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

    namespace
    {

        void validate_mse_loss_inputs(
            const Tensor &prediction,
            const Tensor &target,
            const Tensor &result,
            Reduction reduction)
        {
            if (prediction.device().type() != target.device().type() ||
                prediction.device().type() != result.device().type())
            {
                throw std::runtime_error(
                    "mse_loss expects tensors on the same device");
            }

            if (prediction.dtype() != target.dtype() ||
                prediction.dtype() != result.dtype())
            {
                throw std::runtime_error(
                    "mse_loss expects tensors with the same dtype");
            }

            if (prediction.dtype() != DType::Float32)
            {
                throw std::runtime_error(
                    "mse_loss currently supports only Float32 tensors");
            }

            if (prediction.shape() != target.shape())
            {
                throw std::runtime_error(
                    "mse_loss expects prediction and target with the same shape");
            }

            if (prediction.numel() == 0)
            {
                throw std::runtime_error(
                    "mse_loss expects non-empty tensors");
            }

            if (prediction.storage() != Storage::RowMajor ||
                target.storage() != Storage::RowMajor ||
                result.storage() != Storage::RowMajor)
            {
                throw std::runtime_error(
                    "mse_loss currently supports only RowMajor tensors");
            }

            if (result.rank() != 1 ||
                result.shape()[0] != 1)
            {
                throw std::runtime_error(
                    "mse_loss result must have shape {1}");
            }

            switch (reduction)
            {
            case Reduction::Mean:
            case Reduction::Sum:
                return;

            default:
                throw std::runtime_error(
                    "unknown Reduction in mse_loss");
            }
        }

    }

    void mse_loss(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &result,
        Reduction reduction)
    {
        validate_mse_loss_inputs(
            prediction,
            target,
            result,
            reduction);

        switch (prediction.device().type())
        {
        case DeviceType::CPU:
            mse_loss_cpu_float32(
                prediction,
                target,
                result,
                reduction);
            return;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            mse_loss_cuda_float32(
                prediction,
                target,
                result,
                reduction);
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
                reduction);
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

}