#ifndef KL_BATCH_HPP
#define KL_BATCH_HPP

#include <core/tensor.hpp>

namespace kl
{

    struct Batch
    {
        Tensor inputs;
        Tensor targets;
    };

}

#endif // KL_BATCH_HPP