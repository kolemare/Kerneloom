#ifndef KL_LAYER_HPP
#define KL_LAYER_HPP

#include <core/tensor.hpp>
#include <core/tensor_pool.hpp>
#include <cnn/network/initializer.hpp>

namespace kl
{

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
    };

}

#endif // KL_LAYER_HPP