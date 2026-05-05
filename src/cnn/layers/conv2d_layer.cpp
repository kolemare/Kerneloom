#include <cnn/layers/conv2d_layer.hpp>

#include <core/layout.hpp>
#include <core/storage.hpp>

#include <stdexcept>

namespace kl
{

    Conv2dLayer::Conv2dLayer(
        std::size_t input_channels,
        std::size_t output_channels,
        std::size_t kernel_height,
        std::size_t kernel_width,
        DType dtype,
        Device device,
        Conv2dOptions options)
        : input_channels_(input_channels),
          output_channels_(output_channels),
          kernel_height_(kernel_height),
          kernel_width_(kernel_width),
          dtype_(dtype),
          device_(device),
          options_(options),
          weights_(
              Shape{output_channels, input_channels, kernel_height, kernel_width},
              dtype,
              device,
              Layout::Unknown,
              Storage::RowMajor)
    {
    }

    Tensor Conv2dLayer::forward(const Tensor &input)
    {
        if (input.rank() != 4)
        {
            throw std::runtime_error("Conv2dLayer::forward expects input shape N x C x H x W");
        }

        if (input.shape()[1] != input_channels_)
        {
            throw std::runtime_error("Conv2dLayer::forward input channels mismatch");
        }

        if (input.dtype() != dtype_)
        {
            throw std::runtime_error("Conv2dLayer::forward dtype mismatch");
        }

        if (input.device().type() != device_.type())
        {
            throw std::runtime_error("Conv2dLayer::forward device mismatch");
        }

        if (options_.stride_h == 0 || options_.stride_w == 0)
        {
            throw std::runtime_error("Conv2dLayer::forward stride must be greater than zero");
        }

        if (options_.dilation_h == 0 || options_.dilation_w == 0)
        {
            throw std::runtime_error("Conv2dLayer::forward dilation must be greater than zero");
        }

        const std::size_t n = input.shape()[0];
        const std::size_t h = input.shape()[2];
        const std::size_t w = input.shape()[3];

        const std::size_t effective_kernel_h =
            options_.dilation_h * (kernel_height_ - 1) + 1;

        const std::size_t effective_kernel_w =
            options_.dilation_w * (kernel_width_ - 1) + 1;

        if (effective_kernel_h > h + 2 * options_.padding_h ||
            effective_kernel_w > w + 2 * options_.padding_w)
        {
            throw std::runtime_error("Conv2dLayer::forward effective kernel cannot be larger than padded input");
        }

        const std::size_t output_height =
            (h + 2 * options_.padding_h - effective_kernel_h) / options_.stride_h + 1;

        const std::size_t output_width =
            (w + 2 * options_.padding_w - effective_kernel_w) / options_.stride_w + 1;

        last_input_shape_ = input.shape();
        has_last_input_shape_ = true;

        return Tensor(
            Shape{n, output_channels_, output_height, output_width},
            dtype_,
            device_,
            Layout::NCHW,
            Storage::RowMajor);
    }

    Tensor Conv2dLayer::backward(const Tensor &grad_output)
    {
        if (!has_last_input_shape_)
        {
            throw std::runtime_error("Conv2dLayer::backward called before forward");
        }

        return Tensor(
            last_input_shape_,
            dtype_,
            device_,
            Layout::NCHW,
            Storage::RowMajor);
    }

    const Tensor &Conv2dLayer::weights() const
    {
        return weights_;
    }

    const Conv2dOptions &Conv2dLayer::options() const
    {
        return options_;
    }

}