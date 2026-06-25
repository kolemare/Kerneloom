#include <ops/validation/backward_maxpool2d_validation.hpp>

#include <core/dtype.hpp>
#include <core/storage.hpp>

#include <stdexcept>

namespace kl
{

    void validate_backward_maxpool2d_inputs(
        const Tensor &indices,
        const Tensor &grad_output,
        const Tensor &grad_input)
    {
        if (indices.device().type() != grad_output.device().type() ||
            indices.device().type() != grad_input.device().type())
        {
            throw std::runtime_error(
                "backward_maxpool2d expects tensors on the same device");
        }

        if (grad_output.dtype() !=
            grad_input.dtype())
        {
            throw std::runtime_error(
                "backward_maxpool2d expects grad_output and grad_input with the same dtype");
        }

        if (grad_output.dtype() !=
            DType::Float32)
        {
            throw std::runtime_error(
                "backward_maxpool2d currently supports only Float32 gradients");
        }

        if (indices.dtype() !=
            DType::Int32)
        {
            throw std::runtime_error(
                "backward_maxpool2d expects Int32 indices tensor");
        }

        if (indices.rank() != 4)
        {
            throw std::runtime_error(
                "backward_maxpool2d expects indices shape N x C x OH x OW");
        }

        if (grad_output.rank() != 4)
        {
            throw std::runtime_error(
                "backward_maxpool2d expects grad_output shape N x C x OH x OW");
        }

        if (grad_input.rank() != 4)
        {
            throw std::runtime_error(
                "backward_maxpool2d expects grad_input shape N x C x H x W");
        }

        if (indices.shape() !=
            grad_output.shape())
        {
            throw std::runtime_error(
                "backward_maxpool2d indices shape must match grad_output shape");
        }

        if (indices.shape()[0] != grad_input.shape()[0] ||
            indices.shape()[1] != grad_input.shape()[1])
        {
            throw std::runtime_error(
                "backward_maxpool2d N/C shape mismatch");
        }

        if (indices.storage() != Storage::RowMajor ||
            grad_output.storage() != Storage::RowMajor ||
            grad_input.storage() != Storage::RowMajor)
        {
            throw std::runtime_error(
                "backward_maxpool2d currently supports only RowMajor tensors");
        }
    }

}