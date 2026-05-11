#ifndef KL_MAXPOOL2D_LAYER_HPP
#define KL_MAXPOOL2D_LAYER_HPP

#include <cnn/layers/layer.hpp>
#include <cnn/options/pooling2d_options.hpp>

#include <core/tensor.hpp>
#include <core/tensor_pool.hpp>

namespace kl
{

    class MaxPool2dLayer final : public Layer
    {
    public:
        explicit MaxPool2dLayer(Pooling2dOptions options = {});

        bool verify() const override;

        Tensor &forward(
            Tensor &input,
            TensorPool &pool) override;

        Tensor &backward(
            Tensor &grad_output,
            TensorPool &pool) override;

        const Pooling2dOptions &options() const;

    private:
        Pooling2dOptions options_;

        const Tensor *last_input_ = nullptr;
    };

}

#endif // KL_MAXPOOL2D_LAYER_HPP