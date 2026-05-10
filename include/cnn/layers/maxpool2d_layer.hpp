#ifndef KL_MAXPOOL2D_LAYER_HPP
#define KL_MAXPOOL2D_LAYER_HPP

#include <cnn/layers/layer.hpp>
#include <cnn/options/pooling2d_options.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/layout.hpp>
#include <core/shape.hpp>
#include <core/storage.hpp>
#include <core/tensor.hpp>

namespace kl
{

    class MaxPool2dLayer final : public Layer
    {
    public:
        explicit MaxPool2dLayer(Pooling2dOptions options = {});

        bool verify() const override;

        Tensor forward(const Tensor &input) override;
        Tensor backward(const Tensor &grad_output) override;

        const Pooling2dOptions &options() const;

    private:
        Pooling2dOptions options_;

        Shape last_input_shape_;
        DType last_dtype_;
        Device last_device_;
        Layout last_layout_;
        Storage last_storage_;

        bool has_last_input_ = false;
    };

}

#endif // KL_MAXPOOL2D_LAYER_HPP