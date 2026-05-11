#include <cnn/layers/maxpool2d_layer.hpp>

#include <core/layout.hpp>
#include <core/storage.hpp>

#include <ops/maxpool2d.hpp>

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

    MaxPool2dLayer::MaxPool2dLayer(Pooling2dOptions options)
        : options_(options)
    {
    }

    bool MaxPool2dLayer::verify() const
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

    Tensor &MaxPool2dLayer::forward(
        Tensor &input,
        TensorPool &pool)
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

        Tensor &result = pool.request(
            Shape{n, c, output_h, output_w},
            input.dtype(),
            input.device(),
            Layout::NCHW,
            Storage::RowMajor);

        maxpool2d(input, result, options_);

        last_input_ = &input;

        return result;
    }

    Tensor &MaxPool2dLayer::backward(
        Tensor &grad_output,
        TensorPool &pool)
    {
        if (last_input_ == nullptr)
        {
            throw std::runtime_error("MaxPool2dLayer::backward called before forward");
        }

        Tensor &grad_input = pool.request(
            last_input_->shape(),
            grad_output.dtype(),
            grad_output.device(),
            grad_output.layout(),
            grad_output.storage());

        return grad_input;
    }

    const Pooling2dOptions &MaxPool2dLayer::options() const
    {
        return options_;
    }

}