#include <cnn/optimizers/sgd_optimizer.hpp>

#include <ops/sgd_update.hpp>

#include <stdexcept>

namespace kl
{

    namespace
    {

        void validate_parameter(
            const Parameter &parameter)
        {
            if (parameter.value == nullptr || parameter.grad == nullptr)
            {
                throw std::runtime_error("SGDOptimizer received invalid parameter");
            }

            if (parameter.value->shape() != parameter.grad->shape())
            {
                throw std::runtime_error("SGDOptimizer parameter value/grad shape mismatch");
            }

            if (parameter.value->dtype() != parameter.grad->dtype())
            {
                throw std::runtime_error("SGDOptimizer parameter value/grad dtype mismatch");
            }

            if (parameter.value->device().type() != parameter.grad->device().type())
            {
                throw std::runtime_error("SGDOptimizer parameter value/grad device mismatch");
            }
        }

    }

    SGDOptimizer::SGDOptimizer(float learning_rate)
        : learning_rate_(learning_rate)
    {
        if (learning_rate_ <= 0.0f)
        {
            throw std::runtime_error("SGDOptimizer learning rate must be greater than zero");
        }
    }

    void SGDOptimizer::prepare(
        const std::vector<Parameter> &parameters)
    {
        if (parameters.empty())
        {
            throw std::runtime_error("SGDOptimizer received empty parameter list");
        }

        for (const Parameter &parameter : parameters)
        {
            validate_parameter(parameter);
        }

        prepared_ = true;
    }

    void SGDOptimizer::step(
        const std::vector<Parameter> &parameters)
    {
        if (!prepared_)
        {
            throw std::runtime_error("SGDOptimizer::step called before prepare");
        }

        for (const Parameter &parameter : parameters)
        {
            validate_parameter(parameter);

            sgd_update_unchecked(
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