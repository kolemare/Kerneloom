#ifndef KL_EVALUATION_HPP
#define KL_EVALUATION_HPP

#include <cnn/evaluation/evaluation_result.hpp>
#include <cnn/losses/loss.hpp>
#include <cnn/network/sequential.hpp>

#include <core/tensor_pool.hpp>

#include <data/data_loader.hpp>

namespace kl
{

    class Evaluation
    {
    public:
        Evaluation(
            Sequential &model,
            Loss &loss);

        EvaluationResult evaluate(
            DataLoader &loader);

    private:
        Sequential &model_;
        Loss &loss_;

        TensorPool loss_pool_;
    };

}

#endif // KL_EVALUATION_HPP