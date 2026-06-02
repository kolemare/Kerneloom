#ifndef KL_INFERENCE_HPP
#define KL_INFERENCE_HPP

#include <cnn/network/sequential.hpp>

#include <data/batch.hpp>
#include <data/data_loader.hpp>

#include <core/tensor.hpp>

#include <functional>

namespace kl
{

    class Inference
    {
    public:
        explicit Inference(
            Sequential &model);

        Tensor &forward(
            Batch &batch);

        void run(
            DataLoader &loader,
            const std::function<
                void(
                    const Tensor &prediction,
                    const Tensor &targets)>
                &callback);

    private:
        Sequential &model_;
    };

}

#endif // KL_INFERENCE_HPP