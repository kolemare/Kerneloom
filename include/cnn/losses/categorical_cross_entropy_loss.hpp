#ifndef KL_CATEGORICAL_CROSS_ENTROPY_LOSS_HPP
#define KL_CATEGORICAL_CROSS_ENTROPY_LOSS_HPP

#include <cnn/losses/loss.hpp>
#include <cnn/losses/reduction.hpp>

#include <cstddef>

namespace kl
{

    class CategoricalCrossEntropyLoss final
        : public Loss
    {
    public:
        explicit CategoricalCrossEntropyLoss(
            Reduction reduction =
                Reduction::Mean);

        Tensor &forward(
            const Tensor &prediction,
            const Tensor &target,
            TensorPool &pool) override;

        Tensor &forward(
            const Tensor &prediction,
            const Tensor &target,
            TensorPool &pool,
            std::size_t valid_sample_count) override;

        Tensor &backward(
            TensorPool &pool) override;

        Reduction reduction() const;

    private:
        Reduction reduction_;

        const Tensor *last_prediction_ =
            nullptr;

        const Tensor *last_target_ =
            nullptr;

        std::size_t last_valid_sample_count_ =
            0;
    };

}

#endif // KL_CATEGORICAL_CROSS_ENTROPY_LOSS_HPP