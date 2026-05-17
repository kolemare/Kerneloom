#ifndef KL_ACTIVATION_LAYER_HPP
#define KL_ACTIVATION_LAYER_HPP

#include <cnn/layers/layer.hpp>

#include <core/tensor.hpp>
#include <core/tensor_pool.hpp>

#include <ops/activation.hpp>

namespace kl
{

    class ActivationLayer final : public Layer
    {
    public:
        explicit ActivationLayer(ActivationType activation_type);

        void initialize(const InitializerType &type) override;

        bool verify() const override;

        Tensor &forward(
            Tensor &input,
            TensorPool &pool) override;

        Tensor &backward(
            Tensor &grad_output,
            TensorPool &pool) override;

        ActivationType activation_type() const;

    private:
        ActivationType activation_type_;

        const Tensor *last_input_ = nullptr;
    };

}

#endif // KL_ACTIVATION_LAYER_HPP