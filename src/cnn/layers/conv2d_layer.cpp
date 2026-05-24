#include <cnn/layers/conv2d_layer.hpp>

#include <core/layout.hpp>
#include <core/storage.hpp>

#include <ops/backward_conv2d.hpp>
#include <ops/conv2d.hpp>

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
              Layout::NCHW,
              Storage::RowMajor)
    {
        if (options_.use_bias)
        {
            bias_ = std::make_unique<Tensor>(
                Shape{output_channels},
                dtype,
                device,
                Layout::Unknown,
                Storage::RowMajor);
        }
    }

    void Conv2dLayer::initializeBiases(const InitializerType &type)
    {
        if (bias_ != nullptr)
        {
            Initializer::initialize(*bias_, type);
        }
    }

    void Conv2dLayer::initializeWeights(const InitializerType &type)
    {
        Initializer::initialize(weights_, type);
    }

    void Conv2dLayer::prepareTraining()
    {
        mode_ = LayerMode::Training;

        if (grad_weights_ == nullptr)
        {
            grad_weights_ = std::make_unique<Tensor>(
                weights_.shape(),
                weights_.dtype(),
                weights_.device(),
                weights_.layout(),
                weights_.storage());
        }

        if (bias_ != nullptr && grad_bias_ == nullptr)
        {
            grad_bias_ = std::make_unique<Tensor>(
                bias_->shape(),
                bias_->dtype(),
                bias_->device(),
                bias_->layout(),
                bias_->storage());
        }
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
            if (bias_ == nullptr)
            {
                return false;
            }

            if (bias_->rank() != 1 ||
                bias_->shape()[0] != output_channels_ ||
                bias_->dtype() != dtype_ ||
                bias_->device().type() != device_.type() ||
                bias_->storage() != Storage::RowMajor)
            {
                return false;
            }
        }

        if (!options_.use_bias && bias_ != nullptr)
        {
            return false;
        }

        return true;
    }

    Shape Conv2dLayer::output_shape(
        const Shape &input_shape) const
    {
        const std::size_t output_height = output_size(
            input_shape[2],
            kernel_height_,
            options_.padding_h,
            options_.stride_h,
            options_.dilation_h);

        const std::size_t output_width = output_size(
            input_shape[3],
            kernel_width_,
            options_.padding_w,
            options_.stride_w,
            options_.dilation_w);

        return Shape{
            input_shape[0],
            output_channels_,
            output_height,
            output_width};
    }

    Tensor &Conv2dLayer::forward(
        Tensor &input,
        TensorPool &pool)
    {
        Tensor &result = pool.request(
            output_shape(input.shape()),
            dtype_,
            device_,
            Layout::NCHW,
            Storage::RowMajor);

        const Tensor *bias = nullptr;

        if (bias_ != nullptr)
        {
            bias = bias_.get();
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

        if (mode_ != LayerMode::Training)
        {
            throw std::runtime_error("Conv2dLayer::backward called while layer is not in training mode");
        }

        if (grad_weights_ == nullptr)
        {
            throw std::runtime_error("Conv2dLayer::backward called before prepareTraining");
        }

        if (bias_ != nullptr && grad_bias_ == nullptr)
        {
            throw std::runtime_error("Conv2dLayer::backward called before prepareTraining");
        }

        Tensor &grad_input = pool.request(
            last_input_->shape(),
            grad_output.dtype(),
            grad_output.device(),
            Layout::NCHW,
            Storage::RowMajor);

        Tensor *grad_bias = nullptr;

        if (grad_bias_ != nullptr)
        {
            grad_bias = grad_bias_.get();
        }

        backward_conv2d(
            *last_input_,
            weights_,
            grad_output,
            grad_input,
            *grad_weights_,
            grad_bias,
            options_);

        return grad_input;
    }

    Tensor &Conv2dLayer::weights()
    {
        return weights_;
    }

    Tensor &Conv2dLayer::bias()
    {
        if (bias_ == nullptr)
        {
            throw std::runtime_error("Conv2dLayer::bias requested but bias is disabled");
        }

        return *bias_;
    }

    Tensor &Conv2dLayer::gradWeights()
    {
        if (grad_weights_ == nullptr)
        {
            throw std::runtime_error("Conv2dLayer::gradWeights called before prepareTraining");
        }

        return *grad_weights_;
    }

    Tensor &Conv2dLayer::gradBias()
    {
        if (bias_ == nullptr)
        {
            throw std::runtime_error("Conv2dLayer::gradBias requested but bias is disabled");
        }

        if (grad_bias_ == nullptr)
        {
            throw std::runtime_error("Conv2dLayer::gradBias called before prepareTraining");
        }

        return *grad_bias_;
    }

    const Conv2dOptions &Conv2dLayer::options() const
    {
        return options_;
    }

    std::size_t Conv2dLayer::output_size(
        std::size_t input_size,
        std::size_t kernel_size,
        std::size_t padding,
        std::size_t stride,
        std::size_t dilation) const
    {
        const std::size_t effective_kernel =
            dilation * (kernel_size - 1) + 1;

        return (input_size + 2 * padding - effective_kernel) / stride + 1;
    }

}