#ifndef KL_SGD_OPTIMIZER_HPP
#define KL_SGD_OPTIMIZER_HPP

#include <cnn/optimizers/parameter.hpp>

#include <vector>

namespace kl
{

    class SGDOptimizer
    {
    public:
        explicit SGDOptimizer(float learning_rate);

        void step(
            const std::vector<Parameter> &parameters);

        float learning_rate() const;

    private:
        float learning_rate_;
    };

}

#endif // KL_SGD_OPTIMIZER_HPP