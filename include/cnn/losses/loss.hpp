#ifndef KL_LOSS_HPP
#define KL_LOSS_HPP

#include <core/tensor.hpp>
#include <core/tensor_pool.hpp>

namespace kl
{

    class Loss
    {
    public:
        virtual ~Loss();

        virtual Tensor &forward(
            const Tensor &prediction,
            const Tensor &target,
            TensorPool &pool) = 0;

        virtual Tensor &backward(
            TensorPool &pool) = 0;
    };

}

#endif // KL_LOSS_HPP