#include <cnn/layers/activation_layer.hpp>

#include <stdexcept>

namespace kl
{

    ActivationLayer::ActivationLayer(ActivationType activation_type)
        : activation_type_(activation_type)
    {
    }

    void ActivationLayer::initializeBiases(const InitializerType &type)
    {
        (void)type;
    }

    void ActivationLayer::initializeWeights(const InitializerType &type)
    {
        (void)type;
    }

    bool ActivationLayer::verify() const
    {
        switch (activation_type_)
        {
        case ActivationType::ReLU:
        case ActivationType::Sigmoid:
        case ActivationType::Tanh:
            return true;

        default:
            return false;
        }
    }

    Shape ActivationLayer::output_shape(
        const Shape &input_shape) const
    {
        return input_shape;
    }

    Tensor &ActivationLayer::forward(
        Tensor &input,
        TensorPool &pool)
    {
        (void)pool;

        activation(input, activation_type_);

        last_input_ = &input;

        return input;
    }

    Tensor &ActivationLayer::backward(
        Tensor &grad_output,
        TensorPool &pool)
    {
        if (last_input_ == nullptr)
        {
            throw std::runtime_error("ActivationLayer::backward called before forward");
        }

        Tensor &grad_input = pool.request(
            grad_output.shape(),
            grad_output.dtype(),
            grad_output.device(),
            grad_output.layout(),
            grad_output.storage());

        return grad_input;
    }

    ActivationType ActivationLayer::activation_type() const
    {
        return activation_type_;
    }

}