#ifndef KL_FLATTEN_LAYER_HPP
#define KL_FLATTEN_LAYER_HPP

#include <cnn/layers/layer.hpp>

#include <core/shape.hpp>
#include <core/tensor.hpp>
#include <core/tensor_pool.hpp>

namespace kl
{

    class FlattenLayer final : public Layer
    {
    public:
        FlattenLayer() = default;

        void initialize(const InitializerType &type) override;

        bool verify() const override;

        Tensor &forward(
            Tensor &input,
            TensorPool &pool) override;

        Tensor &backward(
            Tensor &grad_output,
            TensorPool &pool) override;

    private:
        Shape last_input_shape_;
        bool has_last_input_shape_ = false;
    };

}

#endif // KL_FLATTEN_LAYER_HPP