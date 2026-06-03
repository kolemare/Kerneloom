#ifndef KL_BATCH_HPP
#define KL_BATCH_HPP

#include <data/internal/batch_storage.hpp>

#include <cstddef>
#include <memory>

namespace kl
{

    class Batch
    {
    public:
        Batch() = default;

        explicit Batch(
            std::shared_ptr<BatchStorage> storage);

        Batch(
            Tensor inputs,
            Tensor targets);

        Batch(
            Tensor inputs,
            Tensor targets,
            std::size_t valid_sample_count);

        Tensor &inputs();
        const Tensor &inputs() const;

        Tensor &targets();
        const Tensor &targets() const;

        std::size_t valid_sample_count() const;

        bool empty() const;

    private:
        std::shared_ptr<BatchStorage>
            storage_;
    };

}

#endif // KL_BATCH_HPP