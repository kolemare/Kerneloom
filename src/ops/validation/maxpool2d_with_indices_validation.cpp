#include <ops/validation/maxpool2d_with_indices_validation.hpp>

#include <core/dtype.hpp>
#include <core/layout.hpp>
#include <core/storage.hpp>

#include <stdexcept>

namespace kl
{

    namespace
    {

        std::size_t pooling2d_output_size(
            std::size_t input_size,
            std::size_t kernel_size,
            std::size_t padding,
            std::size_t stride)
        {
            return (input_size + 2 * padding - kernel_size) / stride + 1;
        }

    }

    void validate_maxpool2d_with_indices_inputs(
        const Tensor &input,
        const Tensor &result,
        const Tensor &indices,
        const Pooling2dOptions &options)
    {
        if (options.kernel_h == 0 ||
            options.kernel_w == 0)
        {
            throw std::runtime_error(
                "maxpool2d_with_indices kernel size must be greater than zero");
        }

        if (options.stride_h == 0 ||
            options.stride_w == 0)
        {
            throw std::runtime_error(
                "maxpool2d_with_indices stride must be greater than zero");
        }

        if (input.device().type() != result.device().type() ||
            input.device().type() != indices.device().type())
        {
            throw std::runtime_error(
                "maxpool2d_with_indices expects tensors on the same device");
        }

        if (input.dtype() !=
            result.dtype())
        {
            throw std::runtime_error(
                "maxpool2d_with_indices expects input and result with the same dtype");
        }

        if (input.dtype() !=
            DType::Float32)
        {
            throw std::runtime_error(
                "maxpool2d_with_indices currently supports only Float32 tensors");
        }

        if (indices.dtype() !=
            DType::Int32)
        {
            throw std::runtime_error(
                "maxpool2d_with_indices expects Int32 indices tensor");
        }

        if (input.rank() != 4)
        {
            throw std::runtime_error(
                "maxpool2d_with_indices expects input shape N x C x H x W");
        }

        if (result.rank() != 4)
        {
            throw std::runtime_error(
                "maxpool2d_with_indices expects result shape N x C x OH x OW");
        }

        if (indices.rank() != 4)
        {
            throw std::runtime_error(
                "maxpool2d_with_indices expects indices shape N x C x OH x OW");
        }

        if (input.storage() != Storage::RowMajor ||
            result.storage() != Storage::RowMajor ||
            indices.storage() != Storage::RowMajor)
        {
            throw std::runtime_error(
                "maxpool2d_with_indices currently supports only RowMajor tensors");
        }

        if (input.layout() != Layout::NCHW ||
            result.layout() != Layout::NCHW ||
            indices.layout() != Layout::NCHW)
        {
            throw std::runtime_error(
                "maxpool2d_with_indices currently supports only NCHW layout");
        }

        const std::size_t n =
            input.shape()[0];

        const std::size_t c =
            input.shape()[1];

        const std::size_t h =
            input.shape()[2];

        const std::size_t w =
            input.shape()[3];

        if (options.kernel_h >
                h + 2 * options.padding_h ||
            options.kernel_w >
                w + 2 * options.padding_w)
        {
            throw std::runtime_error(
                "maxpool2d_with_indices kernel cannot be larger than padded input");
        }

        const std::size_t output_h =
            pooling2d_output_size(
                h,
                options.kernel_h,
                options.padding_h,
                options.stride_h);

        const std::size_t output_w =
            pooling2d_output_size(
                w,
                options.kernel_w,
                options.padding_w,
                options.stride_w);

        if (result.shape()[0] != n ||
            result.shape()[1] != c ||
            result.shape()[2] != output_h ||
            result.shape()[3] != output_w)
        {
            throw std::runtime_error(
                "maxpool2d_with_indices result tensor has incorrect shape");
        }

        if (indices.shape() !=
            result.shape())
        {
            throw std::runtime_error(
                "maxpool2d_with_indices indices shape must match result shape");
        }
    }

}