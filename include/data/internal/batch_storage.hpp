#ifndef KL_BATCH_STORAGE_HPP
#define KL_BATCH_STORAGE_HPP

#include <core/tensor.hpp>

#include <cstddef>

namespace kl
{

    struct BatchStorage
    {
        Tensor inputs;
        Tensor targets;

        std::size_t valid_sample_count =
            0;
    };

}

#endif // KL_BATCH_STORAGE_HPP