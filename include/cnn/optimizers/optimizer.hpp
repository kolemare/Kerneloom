#ifndef KL_OPTIMIZER_HPP
#define KL_OPTIMIZER_HPP

#include <cnn/optimizers/parameter.hpp>

#include <vector>

namespace kl
{

    class Optimizer
    {
    public:
        virtual ~Optimizer();

        virtual void prepare(
            const std::vector<Parameter> &parameters) = 0;

        virtual void step(
            const std::vector<Parameter> &parameters) = 0;
    };

}

#endif // KL_OPTIMIZER_HPP