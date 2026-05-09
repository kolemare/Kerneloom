#ifndef KL_ACTIVATION_LAYER_HPP
#define KL_ACTIVATION_LAYER_HPP

#include <cnn/layers/layer.hpp>

#include <core/tensor.hpp>

namespace kl
{

    enum class ActivationType
    {
        ReLU,
        Sigmoid,
        Tanh
    };

    const char *activation_type_name(ActivationType type);

    class ActivationLayer final : public Layer
    {
    public:
        explicit ActivationLayer(ActivationType activation_type);

        Tensor forward(const Tensor &input) override;
        Tensor backward(const Tensor &grad_output) override;

        ActivationType activation_type() const;

    private:
        ActivationType activation_type_;

        Shape last_input_shape_;
        DType last_dtype_;
        Device last_device_;
        Layout last_layout_;
        Storage last_storage_;

        bool has_last_input_ = false;
    };

}

#endif // KL_ACTIVATION_LAYER_HPP