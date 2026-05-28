#ifndef KL_SGD_OPTIMIZER_HPP
#define KL_SGD_OPTIMIZER_HPP

#include <cnn/optimizers/optimizer.hpp>

#include <vector>

namespace kl
{

    class SGDOptimizer final : public Optimizer
    {
    public:
        explicit SGDOptimizer(float learning_rate);

        void prepare(
            const std::vector<Parameter> &parameters) override;

        void step(
            const std::vector<Parameter> &parameters) override;

        float learning_rate() const;

    private:
        float learning_rate_;
        bool prepared_ = false;
    };

}

#endif // KL_SGD_OPTIMIZER_HPP