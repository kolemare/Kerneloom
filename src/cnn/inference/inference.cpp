#include <cnn/inference/inference.hpp>

namespace kl
{

    Inference::Inference(
        Sequential &model)
        : model_(
              model)
    {
        model_.prepareInference();
    }

    Tensor &Inference::forward(
        Batch &batch)
    {
        model_.prepareInference();
        model_.reset();

        return model_.forward(
            batch.inputs());
    }

    void Inference::run(
        DataLoader &loader,
        const std::function<
            void(
                const Tensor &prediction,
                const Tensor &targets)>
            &callback)
    {
        while (loader.has_next())
        {
            Batch batch =
                loader.next();

            Tensor &prediction =
                forward(
                    batch);

            callback(
                prediction,
                batch.targets());
        }
    }

}