#include <cnn/layers/activation_layer.hpp>

#include <stdexcept>

namespace kl
{

    ActivationLayer::ActivationLayer(ActivationType activation_type)
        : activation_type_(activation_type),
          last_dtype_(DType::Float32),
          last_device_(Device::cpu()),
          last_layout_(Layout::Unknown),
          last_storage_(Storage::RowMajor)
    {
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

    Tensor ActivationLayer::forward(const Tensor &input)
    {
        last_input_shape_ = input.shape();
        last_dtype_ = input.dtype();
        last_device_ = input.device();
        last_layout_ = input.layout();
        last_storage_ = input.storage();
        has_last_input_ = true;

        Tensor result = input.to(input.device());

        activation(result, activation_type_);

        return result;
    }

    Tensor ActivationLayer::backward(const Tensor &grad_output)
    {
        if (!has_last_input_)
        {
            throw std::runtime_error("ActivationLayer::backward called before forward");
        }

        return Tensor(
            grad_output.shape(),
            grad_output.dtype(),
            grad_output.device(),
            grad_output.layout(),
            grad_output.storage());
    }

    ActivationType ActivationLayer::activation_type() const
    {
        return activation_type_;
    }

}