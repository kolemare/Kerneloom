#ifndef KL_FLATTEN_LAYER_HPP
#define KL_FLATTEN_LAYER_HPP

#include <cnn/layers/layer.hpp>
#include <cnn/layers/cache/layer_cache_key.hpp>

#include <core/shape.hpp>
#include <core/tensor.hpp>
#include <core/tensor_pool.hpp>

namespace kl
{

    class FlattenLayer final : public Layer
    {
    public:
        FlattenLayer() = default;

        void initializeBiases(const InitializerType &type) override;
        void initializeWeights(const InitializerType &type) override;
        void prepareTraining() override;

        bool verify() const override;

        Shape output_shape(
            const Shape &input_shape) const override;

        Tensor &forward(
            Tensor &input,
            TensorPool &pool) override;

        Tensor &backward(
            Tensor &grad_output,
            TensorPool &pool) override;

    private:
        void prepare_cache(
            const Tensor &input);

    private:
        LayerCacheKey cache_key_;
        Shape cached_output_shape_;

        Shape last_input_shape_;
        bool has_last_input_shape_ = false;
    };

}

#endif // KL_FLATTEN_LAYER_HPP