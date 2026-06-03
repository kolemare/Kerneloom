#include <ops/binary_cross_entropy.hpp>

#include <kernels/cpu/losses/binary_cross_entropy_cpu_float32.hpp>

#if defined(KL_ENABLE_CUDA)
#include <kernels/cuda/losses/binary_cross_entropy_cuda_float32.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <kernels/rocm/losses/binary_cross_entropy_rocm_float32.hiph>
#endif

#include <stdexcept>

namespace kl
{

    namespace
    {

        void validate_binary_cross_entropy_inputs(
            const Tensor &prediction,
            const Tensor &target,
            const Tensor &result,
            Reduction reduction,
            std::size_t valid_sample_count)
        {
            if (prediction.device().type() !=
                    target.device().type() ||
                prediction.device().type() !=
                    result.device().type())
            {
                throw std::runtime_error(
                    "binary_cross_entropy expects tensors on the same device");
            }

            if (prediction.dtype() !=
                    target.dtype() ||
                prediction.dtype() !=
                    result.dtype())
            {
                throw std::runtime_error(
                    "binary_cross_entropy expects tensors with the same dtype");
            }

            if (prediction.dtype() !=
                DType::Float32)
            {
                throw std::runtime_error(
                    "binary_cross_entropy currently supports only Float32 tensors");
            }

            if (prediction.shape() !=
                target.shape())
            {
                throw std::runtime_error(
                    "binary_cross_entropy expects prediction and target with the same shape");
            }

            if (prediction.rank() == 0 ||
                prediction.numel() == 0)
            {
                throw std::runtime_error(
                    "binary_cross_entropy expects non-empty tensors with a batch dimension");
            }

            if (valid_sample_count == 0 ||
                valid_sample_count >
                    prediction.shape()[0])
            {
                throw std::runtime_error(
                    "binary_cross_entropy valid sample count is out of range");
            }

            if (prediction.storage() !=
                    Storage::RowMajor ||
                target.storage() !=
                    Storage::RowMajor ||
                result.storage() !=
                    Storage::RowMajor)
            {
                throw std::runtime_error(
                    "binary_cross_entropy currently supports only RowMajor tensors");
            }

            if (result.rank() != 1 ||
                result.shape()[0] != 1)
            {
                throw std::runtime_error(
                    "binary_cross_entropy result must have shape {1}");
            }

            switch (reduction)
            {
            case Reduction::Mean:
            case Reduction::Sum:
                return;

            default:
                throw std::runtime_error(
                    "unknown Reduction in binary_cross_entropy");
            }
        }

    }

    void binary_cross_entropy(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &result,
        Reduction reduction)
    {
        binary_cross_entropy(
            prediction,
            target,
            result,
            reduction,
            prediction.shape()[0]);
    }

    void binary_cross_entropy(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &result,
        Reduction reduction,
        std::size_t valid_sample_count)
    {
        validate_binary_cross_entropy_inputs(
            prediction,
            target,
            result,
            reduction,
            valid_sample_count);

        switch (prediction.device().type())
        {
        case DeviceType::CPU:
            binary_cross_entropy_cpu_float32(
                prediction,
                target,
                result,
                reduction,
                valid_sample_count);
            return;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            binary_cross_entropy_cuda_float32(
                prediction,
                target,
                result,
                reduction,
                valid_sample_count);
            return;
#else
            throw std::runtime_error(
                "CUDA binary_cross_entropy requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            binary_cross_entropy_rocm_float32(
                prediction,
                target,
                result,
                reduction,
                valid_sample_count);
            return;
#else
            throw std::runtime_error(
                "ROCm binary_cross_entropy requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error(
                "unknown DeviceType in binary_cross_entropy");
        }
    }

}