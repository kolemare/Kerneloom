#ifndef KL_LOSS_HPP
#define KL_LOSS_HPP

#include <core/tensor.hpp>
#include <core/tensor_pool.hpp>

#include <cstddef>
#include <stdexcept>

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

        virtual Tensor &forward(
            const Tensor &prediction,
            const Tensor &target,
            TensorPool &pool,
            std::size_t valid_sample_count)
        {
            if (valid_sample_count !=
                prediction.shape()[0])
            {
                throw std::runtime_error(
                    "loss does not support padded samples");
            }

            return forward(
                prediction,
                target,
                pool);
        }

        virtual Tensor &backward(
            TensorPool &pool) = 0;
    };

}

#endif // KL_LOSS_HPP