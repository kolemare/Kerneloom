#ifndef KL_LINEAR_LAYER_HPP
#define KL_LINEAR_LAYER_HPP

#include <cnn/layers/layer.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>
#include <core/tensor_pool.hpp>

#include <cstddef>
#include <memory>

namespace kl
{

    class LinearLayer final : public Layer
    {
    public:
        LinearLayer(
            std::size_t input_features,
            std::size_t output_features,
            DType dtype = DType::Float32,
            Device device = Device::cpu(),
            bool use_bias = true);

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

        Tensor &weights();
        Tensor &bias();

        Tensor &gradWeights();
        Tensor &gradBias();

        std::size_t input_features() const;
        std::size_t output_features() const;

        bool use_bias() const;
        bool hasBias() const;

        void collectParameters(
            std::vector<Parameter> &parameters) override;

    private:
        std::size_t input_features_;
        std::size_t output_features_;

        DType dtype_;
        Device device_;
        bool use_bias_;

        Tensor weights_;
        std::unique_ptr<Tensor> bias_;

        std::unique_ptr<Tensor> grad_weights_;
        std::unique_ptr<Tensor> grad_bias_;

        const Tensor *last_input_ = nullptr;
    };

}

#endif // KL_LINEAR_LAYER_HPP