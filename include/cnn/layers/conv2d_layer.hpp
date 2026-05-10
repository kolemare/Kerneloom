#ifndef KL_CONV2D_LAYER_HPP
#define KL_CONV2D_LAYER_HPP

#include <cnn/layers/layer.hpp>
#include <cnn/options/conv2d_options.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>

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

        bool verify() const override;

        Tensor forward(const Tensor &input) override;
        Tensor backward(const Tensor &grad_output) override;

        const Tensor &weights() const;
        const Tensor &bias() const;
        const Conv2dOptions &options() const;

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

        Shape last_input_shape_;
        bool has_last_input_shape_ = false;
    };

}

#endif // KL_CONV2D_LAYER_HPP