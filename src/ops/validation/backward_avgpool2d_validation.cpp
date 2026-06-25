#include <ops/validation/backward_avgpool2d_validation.hpp>

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

    void validate_backward_avgpool2d_inputs(
        const Tensor &grad_output,
        const Tensor &grad_input,
        const Pooling2dOptions &options)
    {
        if (options.kernel_h == 0 ||
            options.kernel_w == 0)
        {
            throw std::runtime_error(
                "backward_avgpool2d kernel size must be greater than zero");
        }

        if (options.stride_h == 0 ||
            options.stride_w == 0)
        {
            throw std::runtime_error(
                "backward_avgpool2d stride must be greater than zero");
        }

        if (grad_output.device().type() !=
            grad_input.device().type())
        {
            throw std::runtime_error(
                "backward_avgpool2d expects tensors on the same device");
        }

        if (grad_output.dtype() !=
            grad_input.dtype())
        {
            throw std::runtime_error(
                "backward_avgpool2d expects tensors with the same dtype");
        }

        if (grad_output.dtype() !=
            DType::Float32)
        {
            throw std::runtime_error(
                "backward_avgpool2d currently supports only Float32 tensors");
        }

        if (grad_output.rank() != 4)
        {
            throw std::runtime_error(
                "backward_avgpool2d expects grad_output shape N x C x OH x OW");
        }

        if (grad_input.rank() != 4)
        {
            throw std::runtime_error(
                "backward_avgpool2d expects grad_input shape N x C x H x W");
        }

        if (grad_output.storage() != Storage::RowMajor ||
            grad_input.storage() != Storage::RowMajor)
        {
            throw std::runtime_error(
                "backward_avgpool2d currently supports only RowMajor tensors");
        }

        if (grad_output.layout() != Layout::NCHW ||
            grad_input.layout() != Layout::NCHW)
        {
            throw std::runtime_error(
                "backward_avgpool2d currently supports only NCHW layout");
        }

        if (grad_output.shape()[0] != grad_input.shape()[0] ||
            grad_output.shape()[1] != grad_input.shape()[1])
        {
            throw std::runtime_error(
                "backward_avgpool2d N/C shape mismatch");
        }

        const std::size_t input_h =
            grad_input.shape()[2];

        const std::size_t input_w =
            grad_input.shape()[3];

        if (options.kernel_h >
                input_h + 2 * options.padding_h ||
            options.kernel_w >
                input_w + 2 * options.padding_w)
        {
            throw std::runtime_error(
                "backward_avgpool2d kernel cannot be larger than padded input");
        }

        const std::size_t expected_output_h =
            pooling2d_output_size(
                input_h,
                options.kernel_h,
                options.padding_h,
                options.stride_h);

        const std::size_t expected_output_w =
            pooling2d_output_size(
                input_w,
                options.kernel_w,
                options.padding_w,
                options.stride_w);

        if (grad_output.shape()[2] != expected_output_h ||
            grad_output.shape()[3] != expected_output_w)
        {
            throw std::runtime_error(
                "backward_avgpool2d grad_output spatial shape mismatch");
        }
    }

}