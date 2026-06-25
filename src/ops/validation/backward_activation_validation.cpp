#include <ops/validation/backward_activation_validation.hpp>

#include <core/dtype.hpp>
#include <core/storage.hpp>

#include <stdexcept>

namespace kl
{

    void validate_backward_activation_inputs(
        const Tensor &activation_output,
        const Tensor &grad,
        ActivationType type)
    {
        if (activation_output.device().type() !=
            grad.device().type())
        {
            throw std::runtime_error(
                "backward_activation expects tensors on the same device");
        }

        if (activation_output.dtype() !=
            grad.dtype())
        {
            throw std::runtime_error(
                "backward_activation expects tensors with the same dtype");
        }

        if (activation_output.dtype() !=
            DType::Float32)
        {
            throw std::runtime_error(
                "backward_activation currently supports only Float32 tensors");
        }

        if (activation_output.storage() !=
                Storage::RowMajor ||
            grad.storage() !=
                Storage::RowMajor)
        {
            throw std::runtime_error(
                "backward_activation currently supports only RowMajor tensors");
        }

        if (activation_output.shape() !=
            grad.shape())
        {
            throw std::runtime_error(
                "backward_activation expects tensors with the same shape");
        }

        if (type == ActivationType::Softmax)
        {
            if (activation_output.rank() != 2)
            {
                throw std::runtime_error(
                    "Softmax backward currently expects tensor shape N x C");
            }

            if (activation_output.shape()[1] == 0)
            {
                throw std::runtime_error(
                    "Softmax backward class count must be greater than zero");
            }
        }
    }

}