#ifndef KL_ACTIVATION_LAYER_HPP
#define KL_ACTIVATION_LAYER_HPP

#include <cnn/layers/layer.hpp>

#include <core/shape.hpp>
#include <core/tensor.hpp>
#include <core/tensor_pool.hpp>

#include <ops/activation.hpp>
#include <ops/backward_activation.hpp>

namespace kl
{

    class ActivationLayer final : public Layer
    {
    public:
        explicit ActivationLayer(ActivationType activation_type);

        void initializeBiases(const InitializerType &type) override;
        void initializeWeights(const InitializerType &type) override;
        void prepareTraining() override;

        bool verify() const override;

        Shape output_shape(
            const Shape &input_shape) const override;

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