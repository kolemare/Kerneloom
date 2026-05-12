#include <cnn/layers/flatten_layer.hpp>

#include <core/layout.hpp>

#include <stdexcept>

namespace kl
{

    bool FlattenLayer::verify() const
    {
        return true;
    }

    Tensor &FlattenLayer::forward(
        Tensor &input,
        TensorPool &pool)
    {
        (void)pool;

        if (input.rank() < 2)
        {
            throw std::runtime_error("FlattenLayer::forward expects rank >= 2");
        }

        const std::size_t batch_size = input.shape()[0];
        const std::size_t features = input.numel() / batch_size;

        last_input_shape_ = input.shape();
        has_last_input_shape_ = true;

        input.reshape_inplace(Shape{batch_size, features});
        input.set_layout(Layout::Unknown);

        return input;
    }

    Tensor &FlattenLayer::backward(
        Tensor &grad_output,
        TensorPool &pool)
    {
        (void)pool;

        if (!has_last_input_shape_)
        {
            throw std::runtime_error("FlattenLayer::backward called before forward");
        }

        grad_output.reshape_inplace(last_input_shape_);

        return grad_output;
    }

}