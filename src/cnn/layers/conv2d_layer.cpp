#include <cnn/layers/conv2d_layer.hpp>

#include <core/layout.hpp>
#include <core/storage.hpp>

#include <ops/conv2d.hpp>

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
              Layout::NCHW,
              Storage::RowMajor),
          bias_(
              Shape{output_channels},
              dtype,
              device,
              Layout::NCHW,
              Storage::RowMajor)
    {
    }

    void Conv2dLayer::initialize(const InitializerType &type)
    {
        Initializer::initialize(weights(), type);
        Initializer::initialize(bias(), type);
    }

    bool Conv2dLayer::verify() const
    {
        if (input_channels_ == 0 || output_channels_ == 0)
        {
            return false;
        }

        if (kernel_height_ == 0 || kernel_width_ == 0)
        {
            return false;
        }

        if (options_.stride_h == 0 || options_.stride_w == 0)
        {
            return false;
        }

        if (options_.dilation_h == 0 || options_.dilation_w == 0)
        {
            return false;
        }

        if (dtype_ != DType::Float32)
        {
            return false;
        }

        if (weights_.rank() != 4)
        {
            return false;
        }

        if (weights_.shape()[0] != output_channels_ ||
            weights_.shape()[1] != input_channels_ ||
            weights_.shape()[2] != kernel_height_ ||
            weights_.shape()[3] != kernel_width_)
        {
            return false;
        }

        if (weights_.dtype() != dtype_ ||
            weights_.device().type() != device_.type() ||
            weights_.storage() != Storage::RowMajor)
        {
            return false;
        }

        if (options_.use_bias)
        {
            if (bias_.rank() != 1 ||
                bias_.shape()[0] != output_channels_ ||
                bias_.dtype() != dtype_ ||
                bias_.device().type() != device_.type() ||
                bias_.storage() != Storage::RowMajor)
            {
                return false;
            }
        }

        return true;
    }

    Tensor &Conv2dLayer::forward(
        Tensor &input,
        TensorPool &pool)
    {
        const std::size_t n = input.shape()[0];
        const std::size_t h = input.shape()[2];
        const std::size_t w = input.shape()[3];

        const std::size_t output_height = conv2d_output_size(
            h,
            kernel_height_,
            options_.padding_h,
            options_.stride_h,
            options_.dilation_h);

        const std::size_t output_width = conv2d_output_size(
            w,
            kernel_width_,
            options_.padding_w,
            options_.stride_w,
            options_.dilation_w);

        Tensor &result = pool.request(
            Shape{n, output_channels_, output_height, output_width},
            dtype_,
            device_,
            Layout::NCHW,
            Storage::RowMajor);

        const Tensor *bias = nullptr;

        if (options_.use_bias)
        {
            bias = &bias_;
        }

        conv2d(
            input,
            weights_,
            bias,
            result,
            options_);

        last_input_ = &input;

        return result;
    }

    Tensor &Conv2dLayer::backward(
        Tensor &grad_output,
        TensorPool &pool)
    {
        if (last_input_ == nullptr)
        {
            throw std::runtime_error("Conv2dLayer::backward called before forward");
        }

        Tensor &grad_input = pool.request(
            last_input_->shape(),
            grad_output.dtype(),
            grad_output.device(),
            grad_output.layout(),
            grad_output.storage());

        return grad_input;
    }

    Tensor &Conv2dLayer::weights()
    {
        return weights_;
    }

    Tensor &Conv2dLayer::bias()
    {
        return bias_;
    }

    const Conv2dOptions &Conv2dLayer::options() const
    {
        return options_;
    }

}