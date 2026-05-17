#include <cnn/layers/avgpool2d_layer.hpp>

#include <core/layout.hpp>
#include <core/storage.hpp>

#include <ops/avgpool2d.hpp>

#include <stdexcept>

namespace kl
{

    AvgPool2dLayer::AvgPool2dLayer(Pooling2dOptions options)
        : options_(options)
    {
    }

    void AvgPool2dLayer::initializeBiases(const InitializerType &type)
    {
        (void)type;
    }

    void AvgPool2dLayer::initializeWeights(const InitializerType &type)
    {
        (void)type;
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

    Shape AvgPool2dLayer::output_shape(
        const Shape &input_shape) const
    {
        const std::size_t output_height = output_size(
            input_shape[2],
            options_.kernel_h,
            options_.padding_h,
            options_.stride_h);

        const std::size_t output_width = output_size(
            input_shape[3],
            options_.kernel_w,
            options_.padding_w,
            options_.stride_w);

        return Shape{
            input_shape[0],
            input_shape[1],
            output_height,
            output_width};
    }

    Tensor &AvgPool2dLayer::forward(
        Tensor &input,
        TensorPool &pool)
    {
        Tensor &result = pool.request(
            output_shape(input.shape()),
            input.dtype(),
            input.device(),
            Layout::NCHW,
            Storage::RowMajor);

        avgpool2d(input, result, options_);

        last_input_ = &input;

        return result;
    }

    Tensor &AvgPool2dLayer::backward(
        Tensor &grad_output,
        TensorPool &pool)
    {
        if (last_input_ == nullptr)
        {
            throw std::runtime_error("AvgPool2dLayer::backward called before forward");
        }

        Tensor &grad_input = pool.request(
            last_input_->shape(),
            grad_output.dtype(),
            grad_output.device(),
            grad_output.layout(),
            grad_output.storage());

        return grad_input;
    }

    const Pooling2dOptions &AvgPool2dLayer::options() const
    {
        return options_;
    }

    std::size_t AvgPool2dLayer::output_size(
        std::size_t input_size,
        std::size_t kernel_size,
        std::size_t padding,
        std::size_t stride) const
    {
        return (input_size + 2 * padding - kernel_size) / stride + 1;
    }

}