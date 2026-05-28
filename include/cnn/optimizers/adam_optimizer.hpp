#ifndef KL_ADAM_OPTIMIZER_HPP
#define KL_ADAM_OPTIMIZER_HPP

#include <cnn/optimizers/optimizer.hpp>

#include <core/tensor.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace kl
{

    class AdamOptimizer final : public Optimizer
    {
    public:
        explicit AdamOptimizer(
            float learning_rate,
            float beta1 = 0.9f,
            float beta2 = 0.999f,
            float epsilon = 1.0e-8f);

        void prepare(
            const std::vector<Parameter> &parameters) override;

        void step(
            const std::vector<Parameter> &parameters) override;

        float learning_rate() const;
        float beta1() const;
        float beta2() const;
        float epsilon() const;
        std::size_t step_count() const;

    private:
        void validate_parameter(
            const Parameter &parameter) const;

    private:
        float learning_rate_;
        float beta1_;
        float beta2_;
        float epsilon_;

        std::size_t step_ = 0;
        bool prepared_ = false;

        std::vector<std::unique_ptr<Tensor>> first_moments_;
        std::vector<std::unique_ptr<Tensor>> second_moments_;
    };

}

#endif // KL_ADAM_OPTIMIZER_HPP