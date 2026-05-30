#include <ops/backward_categorical_cross_entropy.hpp>

#include <kernels/cpu/losses/backward_categorical_cross_entropy_cpu_float32.hpp>

#include <stdexcept>

namespace kl
{

    namespace
    {

        void validate_backward_categorical_cross_entropy_inputs(
            const Tensor &prediction,
            const Tensor &target,
            const Tensor &grad_prediction,
            Reduction reduction)
        {
            if (prediction.device().type() != target.device().type() ||
                prediction.device().type() != grad_prediction.device().type())
            {
                throw std::runtime_error(
                    "backward_categorical_cross_entropy expects tensors on the same device");
            }

            if (prediction.dtype() != DType::Float32)
            {
                throw std::runtime_error(
                    "backward_categorical_cross_entropy expects Float32 prediction");
            }

            if (target.dtype() != DType::Int32)
            {
                throw std::runtime_error(
                    "backward_categorical_cross_entropy expects Int32 target");
            }

            if (grad_prediction.dtype() != DType::Float32)
            {
                throw std::runtime_error(
                    "backward_categorical_cross_entropy expects Float32 grad_prediction");
            }

            if (prediction.rank() != 2)
            {
                throw std::runtime_error(
                    "backward_categorical_cross_entropy expects prediction shape N x C");
            }

            if (target.rank() != 1)
            {
                throw std::runtime_error(
                    "backward_categorical_cross_entropy expects target shape N");
            }

            if (target.shape()[0] != prediction.shape()[0])
            {
                throw std::runtime_error(
                    "backward_categorical_cross_entropy batch size mismatch");
            }

            if (grad_prediction.shape() != prediction.shape())
            {
                throw std::runtime_error(
                    "backward_categorical_cross_entropy expects grad_prediction shape to match prediction");
            }

            if (prediction.storage() != Storage::RowMajor ||
                target.storage() != Storage::RowMajor ||
                grad_prediction.storage() != Storage::RowMajor)
            {
                throw std::runtime_error(
                    "backward_categorical_cross_entropy currently supports only RowMajor tensors");
            }

            switch (reduction)
            {
            case Reduction::Mean:
            case Reduction::Sum:
                return;

            default:
                throw std::runtime_error(
                    "unknown Reduction in backward_categorical_cross_entropy");
            }
        }

    }

    void backward_categorical_cross_entropy(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &grad_prediction,
        Reduction reduction)
    {
        validate_backward_categorical_cross_entropy_inputs(
            prediction,
            target,
            grad_prediction,
            reduction);

        switch (prediction.device().type())
        {
        case DeviceType::CPU:
            backward_categorical_cross_entropy_cpu_float32(
                prediction,
                target,
                grad_prediction,
                reduction);
            return;

        case DeviceType::CUDA:
            throw std::runtime_error(
                "CUDA backward_categorical_cross_entropy is not implemented yet");

        case DeviceType::ROCM:
            throw std::runtime_error(
                "ROCm backward_categorical_cross_entropy is not implemented yet");

        default:
            throw std::runtime_error(
                "unknown DeviceType in backward_categorical_cross_entropy");
        }
    }

}