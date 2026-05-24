#include <cnn/optimizers/sgd_optimizer.hpp>

#include <ops/sgd_update.hpp>

#include <stdexcept>

namespace kl
{

    SGDOptimizer::SGDOptimizer(float learning_rate)
        : learning_rate_(learning_rate)
    {
        if (learning_rate_ <= 0.0f)
        {
            throw std::runtime_error("SGDOptimizer learning rate must be greater than zero");
        }
    }

    void SGDOptimizer::step(
        const std::vector<Parameter> &parameters)
    {
        for (const Parameter &parameter : parameters)
        {
            if (parameter.value == nullptr || parameter.grad == nullptr)
            {
                throw std::runtime_error("SGDOptimizer received invalid parameter");
            }

            sgd_update(
                *parameter.value,
                *parameter.grad,
                learning_rate_);
        }
    }

    float SGDOptimizer::learning_rate() const
    {
        return learning_rate_;
    }

}