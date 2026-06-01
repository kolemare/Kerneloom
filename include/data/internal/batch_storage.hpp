#ifndef KL_BATCH_STORAGE_HPP
#define KL_BATCH_STORAGE_HPP

#include <core/tensor.hpp>

namespace kl
{

    struct BatchStorage
    {
        Tensor inputs;
        Tensor targets;
    };

}

#endif // KL_BATCH_STORAGE_HPP