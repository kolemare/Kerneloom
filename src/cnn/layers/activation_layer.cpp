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

    void ActivationLayer::prepareTraining()
    {
        mode_ = LayerMode::Training;
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
        (void)pool;

        if (last_input_ == nullptr)
        {
            throw std::runtime_error("ActivationLayer::backward called before forward");
        }

        backward_activation(
            *last_input_,
            grad_output,
            activation_type_);

        return grad_output;
    }

    ActivationType ActivationLayer::activation_type() const
    {
        return activation_type_;
    }

}