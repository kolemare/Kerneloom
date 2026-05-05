#ifndef KL_CONV2D_LAYER_HPP
#define KL_CONV2D_LAYER_HPP

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>
#include <cnn/layer.hpp>

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
            Device device = Device::cpu());

        Tensor forward(const Tensor &input) override;
        Tensor backward(const Tensor &grad_output) override;

        const Tensor &weights() const;

    private:
        std::size_t input_channels_;
        std::size_t output_channels_;
        std::size_t kernel_height_;
        std::size_t kernel_width_;

        DType dtype_;
        Device device_;

        Tensor weights_;

        Shape last_input_shape_;
        bool has_last_input_shape_ = false;
    };

}

#endif // KL_CONV2D_LAYER_HPP