#ifndef KL_SEQUENTIAL_HPP
#define KL_SEQUENTIAL_HPP

#include <cnn/layers/layer.hpp>
#include <cnn/network/initializer.hpp>
#include <cnn/options/conv2d_options.hpp>
#include <cnn/options/pooling2d_options.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>
#include <core/tensor_pool.hpp>

#include <ops/activation.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace kl
{

    class Sequential
    {
    public:
        Sequential(
            std::size_t batch_size,
            std::size_t input_channels,
            std::size_t input_height,
            std::size_t input_width,
            DType dtype = DType::Float32,
            Device device = Device::cpu());

        Sequential(const Sequential &) = delete;
        Sequential &operator=(const Sequential &) = delete;

        Sequential(Sequential &&) noexcept = default;
        Sequential &operator=(Sequential &&) noexcept = default;

        void addConvolutionLayer(
            std::size_t output_channels,
            std::size_t kernel_size);

        void addConvolutionLayer(
            std::size_t output_channels,
            std::size_t kernel_height,
            std::size_t kernel_width,
            Conv2dOptions options);

        void addActivationLayer(
            ActivationType activation_type);

        void addMaxPoolingLayer(
            std::size_t kernel_size);

        void addMaxPoolingLayer(
            std::size_t kernel_height,
            std::size_t kernel_width);

        void addFlattenLayer();

        void addFullyConnectedLayer(
            std::size_t output_features,
            bool use_bias = true);

        void initializeWeights(
            const InitializerType &type);

        void initializeBiases(
            const InitializerType &type);

        Tensor &forward(
            Tensor &input);

        bool verify() const;

        const Shape &input_shape() const;
        const Shape &current_shape() const;

        std::size_t layer_count() const;
        std::size_t pooled_tensor_count() const;

        void reset();
        void clear();

    private:
        void addLayer(
            std::unique_ptr<Layer> layer);

    private:
        Shape input_shape_;
        Shape current_shape_;

        DType dtype_;
        Device device_;

        std::vector<std::unique_ptr<Layer>> layers_;
        TensorPool pool_;
    };

}

#endif // KL_SEQUENTIAL_HPP