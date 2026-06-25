#include <ops/validation/linear_validation.hpp>

#include <core/dtype.hpp>
#include <core/storage.hpp>

#include <stdexcept>

namespace kl
{

    void validate_linear_inputs(
        const Tensor &input,
        const Tensor &weights,
        const Tensor *bias,
        const Tensor &result)
    {
        if (input.device().type() != weights.device().type() ||
            input.device().type() != result.device().type())
        {
            throw std::runtime_error(
                "linear expects input, weights, and result on the same device");
        }

        if (bias != nullptr &&
            bias->device().type() != input.device().type())
        {
            throw std::runtime_error(
                "linear expects bias on the same device as input");
        }

        if (input.dtype() != weights.dtype() ||
            input.dtype() != result.dtype())
        {
            throw std::runtime_error(
                "linear expects input, weights, and result with the same dtype");
        }

        if (bias != nullptr &&
            bias->dtype() != input.dtype())
        {
            throw std::runtime_error(
                "linear expects bias with the same dtype as input");
        }

        if (input.dtype() != DType::Float32)
        {
            throw std::runtime_error(
                "linear currently supports only Float32 tensors");
        }

        if (input.rank() != 2)
        {
            throw std::runtime_error(
                "linear expects input shape N x IN");
        }

        if (weights.rank() != 2)
        {
            throw std::runtime_error(
                "linear expects weights shape OUT x IN");
        }

        if (result.rank() != 2)
        {
            throw std::runtime_error(
                "linear expects result shape N x OUT");
        }

        if (bias != nullptr &&
            bias->rank() != 1)
        {
            throw std::runtime_error(
                "linear expects bias shape OUT");
        }

        if (input.storage() != Storage::RowMajor ||
            weights.storage() != Storage::RowMajor ||
            result.storage() != Storage::RowMajor)
        {
            throw std::runtime_error(
                "linear currently supports only RowMajor tensors");
        }

        if (bias != nullptr &&
            bias->storage() != Storage::RowMajor)
        {
            throw std::runtime_error(
                "linear currently supports only RowMajor bias tensor");
        }

        const std::size_t batch_size =
            input.shape()[0];

        const std::size_t input_features =
            input.shape()[1];

        const std::size_t output_features =
            weights.shape()[0];

        const std::size_t weight_input_features =
            weights.shape()[1];

        if (input_features !=
            weight_input_features)
        {
            throw std::runtime_error(
                "linear input feature size mismatch");
        }

        if (result.shape()[0] != batch_size ||
            result.shape()[1] != output_features)
        {
            throw std::runtime_error(
                "linear result tensor has incorrect shape");
        }

        if (bias != nullptr &&
            bias->shape()[0] != output_features)
        {
            throw std::runtime_error(
                "linear bias shape mismatch");
        }
    }

}