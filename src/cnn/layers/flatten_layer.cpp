#include <cnn/layers/flatten_layer.hpp>

#include <core/layout.hpp>

#include <stdexcept>

namespace kl
{

    void FlattenLayer::initializeBiases(const InitializerType &type)
    {
        (void)type;
    }

    void FlattenLayer::initializeWeights(const InitializerType &type)
    {
        (void)type;
    }

    bool FlattenLayer::verify() const
    {
        return true;
    }

    void FlattenLayer::prepareTraining()
    {
        mode_ = LayerMode::Training;
    }

    Shape FlattenLayer::output_shape(
        const Shape &input_shape) const
    {
        const std::size_t batch_size = input_shape[0];
        const std::size_t features = input_shape.numel() / batch_size;

        return Shape{batch_size, features};
    }

    Tensor &FlattenLayer::forward(
        Tensor &input,
        TensorPool &pool)
    {
        (void)pool;

        prepare_cache(
            input);

        last_input_shape_ =
            cache_key_.shape();

        has_last_input_shape_ =
            true;

        input.reshape_inplace(
            cached_output_shape_);

        input.set_layout(
            Layout::Unknown);

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

        grad_output.reshape_inplace(
            last_input_shape_);

        return grad_output;
    }

    void FlattenLayer::prepare_cache(
        const Tensor &input)
    {
        if (cache_key_.matches(input))
        {
            return;
        }

        cached_output_shape_ =
            output_shape(
                input.shape());

        cache_key_.capture(
            input);
    }

}