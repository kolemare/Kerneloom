#ifndef KL_LAYER_HPP
#define KL_LAYER_HPP

#include <core/tensor.hpp>

namespace kl
{

    class Layer
    {
    public:
        virtual ~Layer() = default;

        virtual Tensor forward(const Tensor &input) = 0;
        virtual Tensor backward(const Tensor &grad_output) = 0;
        virtual bool verify() const = 0;
    };

}

#endif // KL_LAYER_HPP