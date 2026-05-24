#ifndef KL_LAYER_HPP
#define KL_LAYER_HPP

#include <cnn/network/initializer.hpp>
#include <cnn/optimizers/parameter.hpp>

#include <core/shape.hpp>
#include <core/tensor.hpp>
#include <core/tensor_pool.hpp>

#include <vector>

namespace kl
{

    enum class LayerMode
    {
        Inference,
        Training
    };

    class Layer
    {
    public:
        virtual ~Layer() = default;

        virtual Tensor &forward(
            Tensor &input,
            TensorPool &pool) = 0;

        virtual Tensor &backward(
            Tensor &grad_output,
            TensorPool &pool) = 0;

        virtual Shape output_shape(
            const Shape &input_shape) const = 0;

        virtual void initializeWeights(
            const InitializerType &type) = 0;

        virtual void initializeBiases(
            const InitializerType &type) = 0;

        virtual void prepareTraining() = 0;

        virtual bool verify() const = 0;

        virtual void collectParameters(
            std::vector<Parameter> &parameters)
        {
            (void)parameters;
        }

        LayerMode mode() const
        {
            return mode_;
        }

    protected:
        LayerMode mode_ = LayerMode::Inference;
    };

}

#endif // KL_LAYER_HPP