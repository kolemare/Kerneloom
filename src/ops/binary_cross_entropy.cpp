#include <ops/binary_cross_entropy.hpp>

#include <kernels/cpu/losses/binary_cross_entropy_cpu_float32.hpp>

#include <stdexcept>

namespace kl
{

    namespace
    {

        void validate_binary_cross_entropy_inputs(
            const Tensor &prediction,
            const Tensor &target,
            const Tensor &result,
            Reduction reduction)
        {
            if (prediction.device().type() != target.device().type() ||
                prediction.device().type() != result.device().type())
            {
                throw std::runtime_error(
                    "binary_cross_entropy expects tensors on the same device");
            }

            if (prediction.dtype() != target.dtype() ||
                prediction.dtype() != result.dtype())
            {
                throw std::runtime_error(
                    "binary_cross_entropy expects tensors with the same dtype");
            }

            if (prediction.dtype() != DType::Float32)
            {
                throw std::runtime_error(
                    "binary_cross_entropy currently supports only Float32 tensors");
            }

            if (prediction.shape() != target.shape())
            {
                throw std::runtime_error(
                    "binary_cross_entropy expects prediction and target with the same shape");
            }

            if (prediction.storage() != Storage::RowMajor ||
                target.storage() != Storage::RowMajor ||
                result.storage() != Storage::RowMajor)
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
        validate_binary_cross_entropy_inputs(
            prediction,
            target,
            result,
            reduction);

        switch (prediction.device().type())
        {
        case DeviceType::CPU:
            binary_cross_entropy_cpu_float32(
                prediction,
                target,
                result,
                reduction);
            return;

        case DeviceType::CUDA:
            throw std::runtime_error(
                "CUDA binary_cross_entropy is not implemented yet");

        case DeviceType::ROCM:
            throw std::runtime_error(
                "ROCm binary_cross_entropy is not implemented yet");

        default:
            throw std::runtime_error(
                "unknown DeviceType in binary_cross_entropy");
        }
    }

}