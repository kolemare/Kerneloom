#include <cnn/layers/avgpool2d_layer.hpp>

#include <ops/avgpool2d.hpp>

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

    AvgPool2dLayer::AvgPool2dLayer(Pooling2dOptions options)
        : options_(options),
          last_dtype_(DType::Float32),
          last_device_(Device::cpu()),
          last_layout_(Layout::Unknown),
          last_storage_(Storage::RowMajor)
    {
    }

    bool AvgPool2dLayer::verify() const
    {
        if (options_.kernel_h == 0 || options_.kernel_w == 0)
        {
            return false;
        }

        if (options_.stride_h == 0 || options_.stride_w == 0)
        {
            return false;
        }

        return true;
    }

    Tensor AvgPool2dLayer::forward(const Tensor &input)
    {
        const std::size_t n = input.shape()[0];
        const std::size_t c = input.shape()[1];
        const std::size_t h = input.shape()[2];
        const std::size_t w = input.shape()[3];

        const std::size_t output_h = pooling2d_output_size(
            h,
            options_.kernel_h,
            options_.padding_h,
            options_.stride_h);

        const std::size_t output_w = pooling2d_output_size(
            w,
            options_.kernel_w,
            options_.padding_w,
            options_.stride_w);

        Tensor result(
            Shape{n, c, output_h, output_w},
            input.dtype(),
            input.device(),
            Layout::NCHW,
            Storage::RowMajor);

        avgpool2d(input, result, options_);

        last_input_shape_ = input.shape();
        last_dtype_ = input.dtype();
        last_device_ = input.device();
        last_layout_ = input.layout();
        last_storage_ = input.storage();
        has_last_input_ = true;

        return result;
    }

    Tensor AvgPool2dLayer::backward(const Tensor &grad_output)
    {
        if (!has_last_input_)
        {
            throw std::runtime_error("AvgPool2dLayer::backward called before forward");
        }

        return Tensor(
            last_input_shape_,
            grad_output.dtype(),
            grad_output.device(),
            grad_output.layout(),
            grad_output.storage());
    }

    const Pooling2dOptions &AvgPool2dLayer::options() const
    {
        return options_;
    }

}