#include <ops/validation/adam_update_validation.hpp>

#include <core/dtype.hpp>
#include <core/storage.hpp>

#include <stdexcept>

namespace kl
{

    void validate_adam_update_inputs(
        const Tensor &value,
        const Tensor &grad,
        const Tensor &first_moment,
        const Tensor &second_moment,
        float learning_rate,
        float beta1,
        float beta2,
        float epsilon,
        float beta1_power,
        float beta2_power)
    {
        if (value.device().type() != grad.device().type() ||
            value.device().type() != first_moment.device().type() ||
            value.device().type() != second_moment.device().type())
        {
            throw std::runtime_error(
                "adam_update expects tensors on the same device");
        }

        if (value.dtype() != grad.dtype() ||
            value.dtype() != first_moment.dtype() ||
            value.dtype() != second_moment.dtype())
        {
            throw std::runtime_error(
                "adam_update expects tensors with the same dtype");
        }

        if (value.dtype() != DType::Float32)
        {
            throw std::runtime_error(
                "adam_update currently supports only Float32 tensors");
        }

        if (value.storage() != Storage::RowMajor ||
            grad.storage() != Storage::RowMajor ||
            first_moment.storage() != Storage::RowMajor ||
            second_moment.storage() != Storage::RowMajor)
        {
            throw std::runtime_error(
                "adam_update currently supports only RowMajor tensors");
        }

        if (value.shape() != grad.shape() ||
            value.shape() != first_moment.shape() ||
            value.shape() != second_moment.shape())
        {
            throw std::runtime_error(
                "adam_update expects tensors with the same shape");
        }

        if (learning_rate <= 0.0f)
        {
            throw std::runtime_error(
                "adam_update learning_rate must be greater than zero");
        }

        if (beta1 <= 0.0f ||
            beta1 >= 1.0f)
        {
            throw std::runtime_error(
                "adam_update beta1 must be in range (0, 1)");
        }

        if (beta2 <= 0.0f ||
            beta2 >= 1.0f)
        {
            throw std::runtime_error(
                "adam_update beta2 must be in range (0, 1)");
        }

        if (epsilon <= 0.0f)
        {
            throw std::runtime_error(
                "adam_update epsilon must be greater than zero");
        }

        if (beta1_power < 0.0f ||
            beta1_power >= 1.0f)
        {
            throw std::runtime_error(
                "adam_update beta1_power must be in range [0, 1)");
        }

        if (beta2_power < 0.0f ||
            beta2_power >= 1.0f)
        {
            throw std::runtime_error(
                "adam_update beta2_power must be in range [0, 1)");
        }
    }

}