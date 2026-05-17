#include <cnn/network/sequential.hpp>

#include <cnn/layers/activation_layer.hpp>
#include <cnn/layers/conv2d_layer.hpp>
#include <cnn/layers/flatten_layer.hpp>
#include <cnn/layers/linear_layer.hpp>
#include <cnn/layers/maxpool2d_layer.hpp>

namespace kl
{

    Sequential::Sequential(
        std::size_t batch_size,
        std::size_t input_channels,
        std::size_t input_height,
        std::size_t input_width,
        DType dtype,
        Device device)
        : input_shape_(Shape{batch_size, input_channels, input_height, input_width}),
          current_shape_(input_shape_),
          dtype_(dtype),
          device_(device)
    {
    }

    void Sequential::addConvolutionLayer(
        std::size_t output_channels,
        std::size_t kernel_size)
    {
        addConvolutionLayer(
            output_channels,
            kernel_size,
            kernel_size,
            Conv2dOptions{});
    }

    void Sequential::addConvolutionLayer(
        std::size_t output_channels,
        std::size_t kernel_height,
        std::size_t kernel_width,
        Conv2dOptions options)
    {
        addLayer(std::make_unique<Conv2dLayer>(
            current_shape_[1],
            output_channels,
            kernel_height,
            kernel_width,
            dtype_,
            device_,
            options));
    }

    void Sequential::addActivationLayer(
        ActivationType activation_type)
    {
        addLayer(std::make_unique<ActivationLayer>(
            activation_type));
    }

    void Sequential::addMaxPoolingLayer(
        std::size_t kernel_size)
    {
        addMaxPoolingLayer(
            kernel_size,
            kernel_size);
    }

    void Sequential::addMaxPoolingLayer(
        std::size_t kernel_height,
        std::size_t kernel_width)
    {
        Pooling2dOptions options;
        options.kernel_h = kernel_height;
        options.kernel_w = kernel_width;
        options.stride_h = kernel_height;
        options.stride_w = kernel_width;
        options.padding_h = 0;
        options.padding_w = 0;

        addLayer(std::make_unique<MaxPool2dLayer>(
            options));
    }

    void Sequential::addFlattenLayer()
    {
        addLayer(std::make_unique<FlattenLayer>());
    }

    void Sequential::addFullyConnectedLayer(
        std::size_t output_features,
        bool use_bias)
    {
        addLayer(std::make_unique<LinearLayer>(
            current_shape_[1],
            output_features,
            dtype_,
            device_,
            use_bias));
    }

    void Sequential::initializeWeights(
        const InitializerType &type)
    {
        for (std::unique_ptr<Layer> &layer : layers_)
        {
            layer->initializeWeights(type);
        }
    }

    void Sequential::initializeBiases(
        const InitializerType &type)
    {
        for (std::unique_ptr<Layer> &layer : layers_)
        {
            layer->initializeBiases(type);
        }
    }

    Tensor &Sequential::forward(
        Tensor &input)
    {
        pool_.reset();

        Tensor *current = &input;

        for (std::unique_ptr<Layer> &layer : layers_)
        {
            current = &layer->forward(*current, pool_);
        }

        return *current;
    }

    bool Sequential::verify() const
    {
        for (const std::unique_ptr<Layer> &layer : layers_)
        {
            if (!layer->verify())
            {
                return false;
            }
        }

        return true;
    }

    const Shape &Sequential::input_shape() const
    {
        return input_shape_;
    }

    const Shape &Sequential::current_shape() const
    {
        return current_shape_;
    }

    std::size_t Sequential::layer_count() const
    {
        return layers_.size();
    }

    std::size_t Sequential::pooled_tensor_count() const
    {
        return pool_.tensor_count();
    }

    void Sequential::reset()
    {
        pool_.reset();
    }

    void Sequential::clear()
    {
        pool_.clear();
    }

    void Sequential::addLayer(
        std::unique_ptr<Layer> layer)
    {
        current_shape_ = layer->output_shape(current_shape_);
        layers_.push_back(std::move(layer));
    }

}