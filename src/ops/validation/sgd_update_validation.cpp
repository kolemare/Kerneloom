#include <ops/validation/sgd_update_validation.hpp>

#include <core/dtype.hpp>
#include <core/storage.hpp>

#include <stdexcept>

namespace kl
{

    void validate_sgd_update_inputs(
        const Tensor &value,
        const Tensor &grad)
    {
        if (value.device().type() !=
            grad.device().type())
        {
            throw std::runtime_error(
                "sgd_update expects tensors on the same device");
        }

        if (value.dtype() !=
            grad.dtype())
        {
            throw std::runtime_error(
                "sgd_update expects tensors with the same dtype");
        }

        if (value.dtype() !=
            DType::Float32)
        {
            throw std::runtime_error(
                "sgd_update currently supports only Float32 tensors");
        }

        if (value.storage() != Storage::RowMajor ||
            grad.storage() != Storage::RowMajor)
        {
            throw std::runtime_error(
                "sgd_update currently supports only RowMajor tensors");
        }

        if (value.shape() !=
            grad.shape())
        {
            throw std::runtime_error(
                "sgd_update expects tensors with the same shape");
        }
    }

}