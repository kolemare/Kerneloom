#include <ops/validation/backward_conv2d_validation.hpp>

#include <core/dtype.hpp>
#include <core/layout.hpp>
#include <core/storage.hpp>

#include <stdexcept>

namespace kl
{

    namespace
    {

        std::size_t conv2d_output_size(
            std::size_t input_size,
            std::size_t kernel_size,
            std::size_t padding,
            std::size_t stride,
            std::size_t dilation)
        {
            const std::size_t effective_kernel =
                dilation * (kernel_size - 1) + 1;

            return (input_size + 2 * padding - effective_kernel) / stride + 1;
        }

    }

    void validate_backward_conv2d_inputs(
        const Tensor &input,
        const Tensor &weights,
        const Tensor &grad_output,
        const Tensor &grad_input,
        const Tensor &grad_weights,
        const Tensor *grad_bias,
        const Conv2dOptions &options)
    {
        if (options.stride_h == 0 ||
            options.stride_w == 0)
        {
            throw std::runtime_error(
                "backward_conv2d stride must be greater than zero");
        }

        if (options.dilation_h == 0 ||
            options.dilation_w == 0)
        {
            throw std::runtime_error(
                "backward_conv2d dilation must be greater than zero");
        }

        if (input.device().type() != weights.device().type() ||
            input.device().type() != grad_output.device().type() ||
            input.device().type() != grad_input.device().type() ||
            input.device().type() != grad_weights.device().type())
        {
            throw std::runtime_error(
                "backward_conv2d expects tensors on the same device");
        }

        if (grad_bias != nullptr &&
            grad_bias->device().type() !=
                input.device().type())
        {
            throw std::runtime_error(
                "backward_conv2d expects grad_bias on the same device");
        }

        if (input.dtype() != weights.dtype() ||
            input.dtype() != grad_output.dtype() ||
            input.dtype() != grad_input.dtype() ||
            input.dtype() != grad_weights.dtype())
        {
            throw std::runtime_error(
                "backward_conv2d expects tensors with the same dtype");
        }

        if (grad_bias != nullptr &&
            grad_bias->dtype() != input.dtype())
        {
            throw std::runtime_error(
                "backward_conv2d expects grad_bias with the same dtype");
        }

        if (input.dtype() != DType::Float32)
        {
            throw std::runtime_error(
                "backward_conv2d currently supports only Float32 tensors");
        }

        if (input.rank() != 4)
        {
            throw std::runtime_error(
                "backward_conv2d expects input shape N x C x H x W");
        }

        if (weights.rank() != 4)
        {
            throw std::runtime_error(
                "backward_conv2d expects weights shape OUT x IN x KH x KW");
        }

        if (grad_output.rank() != 4)
        {
            throw std::runtime_error(
                "backward_conv2d expects grad_output shape N x OUT x OH x OW");
        }

        if (grad_input.rank() != 4)
        {
            throw std::runtime_error(
                "backward_conv2d expects grad_input shape N x C x H x W");
        }

        if (grad_weights.rank() != 4)
        {
            throw std::runtime_error(
                "backward_conv2d expects grad_weights shape OUT x IN x KH x KW");
        }

        if (input.storage() != Storage::RowMajor ||
            weights.storage() != Storage::RowMajor ||
            grad_output.storage() != Storage::RowMajor ||
            grad_input.storage() != Storage::RowMajor ||
            grad_weights.storage() != Storage::RowMajor)
        {
            throw std::runtime_error(
                "backward_conv2d currently supports only RowMajor tensors");
        }

        if (grad_bias != nullptr &&
            grad_bias->storage() != Storage::RowMajor)
        {
            throw std::runtime_error(
                "backward_conv2d currently supports only RowMajor grad_bias");
        }

        if (input.layout() != Layout::NCHW ||
            weights.layout() != Layout::NCHW ||
            grad_output.layout() != Layout::NCHW ||
            grad_input.layout() != Layout::NCHW ||
            grad_weights.layout() != Layout::NCHW)
        {
            throw std::runtime_error(
                "backward_conv2d currently supports only NCHW layout");
        }

        const std::size_t batch_size =
            input.shape()[0];

        const std::size_t input_channels =
            input.shape()[1];

        const std::size_t input_h =
            input.shape()[2];

        const std::size_t input_w =
            input.shape()[3];

        const std::size_t output_channels =
            weights.shape()[0];

        const std::size_t weight_input_channels =
            weights.shape()[1];

        const std::size_t kernel_h =
            weights.shape()[2];

        const std::size_t kernel_w =
            weights.shape()[3];

        if (input_channels !=
            weight_input_channels)
        {
            throw std::runtime_error(
                "backward_conv2d input channel mismatch");
        }

        if (grad_input.shape() !=
            input.shape())
        {
            throw std::runtime_error(
                "backward_conv2d grad_input shape must match input shape");
        }

        if (grad_weights.shape() !=
            weights.shape())
        {
            throw std::runtime_error(
                "backward_conv2d grad_weights shape must match weights shape");
        }

        const std::size_t expected_output_h =
            conv2d_output_size(
                input_h,
                kernel_h,
                options.padding_h,
                options.stride_h,
                options.dilation_h);

        const std::size_t expected_output_w =
            conv2d_output_size(
                input_w,
                kernel_w,
                options.padding_w,
                options.stride_w,
                options.dilation_w);

        if (grad_output.shape()[0] != batch_size ||
            grad_output.shape()[1] != output_channels ||
            grad_output.shape()[2] != expected_output_h ||
            grad_output.shape()[3] != expected_output_w)
        {
            throw std::runtime_error(
                "backward_conv2d grad_output shape mismatch");
        }

        if (grad_bias != nullptr)
        {
            if (grad_bias->rank() != 1 ||
                grad_bias->shape()[0] != output_channels)
            {
                throw std::runtime_error(
                    "backward_conv2d grad_bias shape mismatch");
            }
        }
    }

}