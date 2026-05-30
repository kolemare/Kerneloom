#ifndef KL_BINARY_CROSS_ENTROPY_LOSS_HPP
#define KL_BINARY_CROSS_ENTROPY_LOSS_HPP

#include <cnn/losses/loss.hpp>
#include <cnn/losses/reduction.hpp>

namespace kl
{

    class BinaryCrossEntropyLoss final : public Loss
    {
    public:
        explicit BinaryCrossEntropyLoss(
            Reduction reduction = Reduction::Mean);

        Tensor &forward(
            const Tensor &prediction,
            const Tensor &target,
            TensorPool &pool) override;

        Tensor &backward(
            TensorPool &pool) override;

        Reduction reduction() const;

    private:
        Reduction reduction_;

        const Tensor *last_prediction_ = nullptr;
        const Tensor *last_target_ = nullptr;
    };

}

#endif // KL_BINARY_CROSS_ENTROPY_LOSS_HPP