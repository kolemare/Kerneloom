#ifndef KL_CONV2D_LAYER_HPP
#define KL_CONV2D_LAYER_HPP

#include <cnn/layers/layer.hpp>
#include <cnn/options/conv2d_options.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>
#include <core/tensor_pool.hpp>

#include <cstddef>

namespace kl
{

    class Conv2dLayer final : public Layer
    {
    public:
        Conv2dLayer(
            std::size_t input_channels,
            std::size_t output_channels,
            std::size_t kernel_height,
            std::size_t kernel_width,
            DType dtype = DType::Float32,
            Device device = Device::cpu(),
            Conv2dOptions options = {});

        void initializeBiases(const InitializerType &type) override;
        void initializeWeights(const InitializerType &type) override;

        bool verify() const override;

        Shape output_shape(
            const Shape &input_shape) const override;

        Tensor &forward(
            Tensor &input,
            TensorPool &pool) override;

        Tensor &backward(
            Tensor &grad_output,
            TensorPool &pool) override;

        Tensor &weights();
        Tensor &bias();

        const Conv2dOptions &options() const;

    private:
        std::size_t output_size(
            std::size_t input_size,
            std::size_t kernel_size,
            std::size_t padding,
            std::size_t stride,
            std::size_t dilation) const;

    private:
        std::size_t input_channels_;
        std::size_t output_channels_;
        std::size_t kernel_height_;
        std::size_t kernel_width_;

        DType dtype_;
        Device device_;
        Conv2dOptions options_;

        Tensor weights_;
        Tensor bias_;

        const Tensor *last_input_ = nullptr;
    };

}

#endif // KL_CONV2D_LAYER_HPP