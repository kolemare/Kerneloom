#ifndef KL_BATCH_HPP
#define KL_BATCH_HPP

#include <data/internal/batch_storage.hpp>

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

        Tensor &inputs();
        const Tensor &inputs() const;

        Tensor &targets();
        const Tensor &targets() const;

        bool empty() const;

    private:
        std::shared_ptr<BatchStorage> storage_;
    };

}

#endif // KL_BATCH_HPP