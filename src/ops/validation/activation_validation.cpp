#include <ops/validation/activation_validation.hpp>

#include <core/dtype.hpp>
#include <core/storage.hpp>

#include <stdexcept>

namespace kl
{

    void validate_activation_tensor(
        const Tensor &tensor,
        ActivationType type)
    {
        if (tensor.dtype() != DType::Float32)
        {
            throw std::runtime_error(
                "activation currently supports only Float32 tensors");
        }

        if (tensor.storage() != Storage::RowMajor)
        {
            throw std::runtime_error(
                "activation currently supports only RowMajor tensors");
        }

        if (type == ActivationType::Softmax)
        {
            if (tensor.rank() != 2)
            {
                throw std::runtime_error(
                    "Softmax activation currently expects tensor shape N x C");
            }

            if (tensor.shape()[1] == 0)
            {
                throw std::runtime_error(
                    "Softmax activation class count must be greater than zero");
            }
        }
    }

}