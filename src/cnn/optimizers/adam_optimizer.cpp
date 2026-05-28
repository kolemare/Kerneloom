#include <cnn/optimizers/adam_optimizer.hpp>

#include <cnn/network/initializer.hpp>

#include <ops/adam_update.hpp>

#include <cmath>
#include <stdexcept>

namespace kl
{

    AdamOptimizer::AdamOptimizer(
        float learning_rate,
        float beta1,
        float beta2,
        float epsilon)
        : learning_rate_(learning_rate),
          beta1_(beta1),
          beta2_(beta2),
          epsilon_(epsilon)
    {
        if (learning_rate_ <= 0.0f)
        {
            throw std::runtime_error("AdamOptimizer learning rate must be greater than zero");
        }

        if (beta1_ <= 0.0f || beta1_ >= 1.0f)
        {
            throw std::runtime_error("AdamOptimizer beta1 must be in range (0, 1)");
        }

        if (beta2_ <= 0.0f || beta2_ >= 1.0f)
        {
            throw std::runtime_error("AdamOptimizer beta2 must be in range (0, 1)");
        }

        if (epsilon_ <= 0.0f)
        {
            throw std::runtime_error("AdamOptimizer epsilon must be greater than zero");
        }
    }

    void AdamOptimizer::prepare(
        const std::vector<Parameter> &parameters)
    {
        if (parameters.empty())
        {
            throw std::runtime_error("AdamOptimizer received empty parameter list");
        }

        first_moments_.clear();
        second_moments_.clear();

        first_moments_.reserve(parameters.size());
        second_moments_.reserve(parameters.size());

        for (const Parameter &parameter : parameters)
        {
            validate_parameter(parameter);

            first_moments_.push_back(
                std::make_unique<Tensor>(
                    parameter.value->shape(),
                    parameter.value->dtype(),
                    parameter.value->device(),
                    parameter.value->layout(),
                    parameter.value->storage()));

            second_moments_.push_back(
                std::make_unique<Tensor>(
                    parameter.value->shape(),
                    parameter.value->dtype(),
                    parameter.value->device(),
                    parameter.value->layout(),
                    parameter.value->storage()));

            Initializer::initialize(
                *first_moments_.back(),
                InitializerType::Zeros);

            Initializer::initialize(
                *second_moments_.back(),
                InitializerType::Zeros);
        }

        step_ = 0;
        prepared_ = true;
    }

    void AdamOptimizer::step(
        const std::vector<Parameter> &parameters)
    {
        if (!prepared_)
        {
            throw std::runtime_error("AdamOptimizer::step called before prepare");
        }

        if (parameters.size() != first_moments_.size() ||
            parameters.size() != second_moments_.size())
        {
            throw std::runtime_error("AdamOptimizer parameter count changed after prepare");
        }

        ++step_;

        const float beta1_power =
            std::pow(beta1_, static_cast<float>(step_));

        const float beta2_power =
            std::pow(beta2_, static_cast<float>(step_));

        for (std::size_t i = 0; i < parameters.size(); ++i)
        {
            const Parameter &parameter = parameters[i];

            validate_parameter(parameter);

            adam_update(
                *parameter.value,
                *parameter.grad,
                *first_moments_[i],
                *second_moments_[i],
                learning_rate_,
                beta1_,
                beta2_,
                epsilon_,
                beta1_power,
                beta2_power);
        }
    }

    float AdamOptimizer::learning_rate() const
    {
        return learning_rate_;
    }

    float AdamOptimizer::beta1() const
    {
        return beta1_;
    }

    float AdamOptimizer::beta2() const
    {
        return beta2_;
    }

    float AdamOptimizer::epsilon() const
    {
        return epsilon_;
    }

    std::size_t AdamOptimizer::step_count() const
    {
        return step_;
    }

    void AdamOptimizer::validate_parameter(
        const Parameter &parameter) const
    {
        if (parameter.value == nullptr || parameter.grad == nullptr)
        {
            throw std::runtime_error("AdamOptimizer received invalid parameter");
        }

        if (parameter.value->shape() != parameter.grad->shape())
        {
            throw std::runtime_error("AdamOptimizer parameter value/grad shape mismatch");
        }

        if (parameter.value->dtype() != parameter.grad->dtype())
        {
            throw std::runtime_error("AdamOptimizer parameter value/grad dtype mismatch");
        }

        if (parameter.value->device().type() != parameter.grad->device().type())
        {
            throw std::runtime_error("AdamOptimizer parameter value/grad device mismatch");
        }
    }

}