#include <ops/validation/maxpool2d_validation.hpp>

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

    void validate_maxpool2d_inputs(
        const Tensor &input,
        const Tensor &result,
        const Pooling2dOptions &options)
    {
        if (options.kernel_h == 0 ||
            options.kernel_w == 0)
        {
            throw std::runtime_error(
                "maxpool2d kernel size must be greater than zero");
        }

        if (options.stride_h == 0 ||
            options.stride_w == 0)
        {
            throw std::runtime_error(
                "maxpool2d stride must be greater than zero");
        }

        if (input.device().type() !=
            result.device().type())
        {
            throw std::runtime_error(
                "maxpool2d expects input and result on the same device");
        }

        if (input.dtype() !=
            result.dtype())
        {
            throw std::runtime_error(
                "maxpool2d expects input and result with the same dtype");
        }

        if (input.dtype() !=
            DType::Float32)
        {
            throw std::runtime_error(
                "maxpool2d currently supports only Float32 tensors");
        }

        if (input.rank() != 4)
        {
            throw std::runtime_error(
                "maxpool2d expects input shape N x C x H x W");
        }

        if (result.rank() != 4)
        {
            throw std::runtime_error(
                "maxpool2d expects result shape N x C x OH x OW");
        }

        if (input.storage() != Storage::RowMajor ||
            result.storage() != Storage::RowMajor)
        {
            throw std::runtime_error(
                "maxpool2d currently supports only RowMajor tensors");
        }

        if (input.layout() != Layout::NCHW ||
            result.layout() != Layout::NCHW)
        {
            throw std::runtime_error(
                "maxpool2d currently supports only NCHW layout");
        }

        const std::size_t N =
            input.shape()[0];

        const std::size_t C =
            input.shape()[1];

        const std::size_t H =
            input.shape()[2];

        const std::size_t W =
            input.shape()[3];

        if (options.kernel_h >
                H + 2 * options.padding_h ||
            options.kernel_w >
                W + 2 * options.padding_w)
        {
            throw std::runtime_error(
                "maxpool2d kernel cannot be larger than padded input");
        }

        const std::size_t OH =
            pooling2d_output_size(
                H,
                options.kernel_h,
                options.padding_h,
                options.stride_h);

        const std::size_t OW =
            pooling2d_output_size(
                W,
                options.kernel_w,
                options.padding_w,
                options.stride_w);

        if (result.shape()[0] != N ||
            result.shape()[1] != C ||
            result.shape()[2] != OH ||
            result.shape()[3] != OW)
        {
            throw std::runtime_error(
                "maxpool2d result tensor has incorrect shape");
        }
    }

}