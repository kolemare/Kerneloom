#include <ops/validation/backward_binary_cross_entropy_validation.hpp>

#include <core/dtype.hpp>
#include <core/storage.hpp>

#include <stdexcept>

namespace kl
{

    void validate_backward_binary_cross_entropy_inputs(
        const Tensor &prediction,
        const Tensor &target,
        const Tensor &grad_prediction,
        Reduction reduction,
        std::size_t valid_sample_count)
    {
        if (prediction.device().type() !=
                target.device().type() ||
            prediction.device().type() !=
                grad_prediction.device().type())
        {
            throw std::runtime_error(
                "backward_binary_cross_entropy expects tensors on the same device");
        }

        if (prediction.dtype() !=
                target.dtype() ||
            prediction.dtype() !=
                grad_prediction.dtype())
        {
            throw std::runtime_error(
                "backward_binary_cross_entropy expects tensors with the same dtype");
        }

        if (prediction.dtype() !=
            DType::Float32)
        {
            throw std::runtime_error(
                "backward_binary_cross_entropy currently supports only Float32 tensors");
        }

        if (prediction.shape() !=
                target.shape() ||
            prediction.shape() !=
                grad_prediction.shape())
        {
            throw std::runtime_error(
                "backward_binary_cross_entropy expects tensors with the same shape");
        }

        if (prediction.rank() == 0 ||
            prediction.numel() == 0)
        {
            throw std::runtime_error(
                "backward_binary_cross_entropy expects non-empty tensors with a batch dimension");
        }

        if (valid_sample_count == 0 ||
            valid_sample_count >
                prediction.shape()[0])
        {
            throw std::runtime_error(
                "backward_binary_cross_entropy valid sample count is out of range");
        }

        if (prediction.storage() !=
                Storage::RowMajor ||
            target.storage() !=
                Storage::RowMajor ||
            grad_prediction.storage() !=
                Storage::RowMajor)
        {
            throw std::runtime_error(
                "backward_binary_cross_entropy currently supports only RowMajor tensors");
        }

        switch (reduction)
        {
        case Reduction::Mean:
        case Reduction::Sum:
            return;

        default:
            throw std::runtime_error(
                "unknown Reduction in backward_binary_cross_entropy");
        }
    }

}