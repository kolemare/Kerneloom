#ifndef KL_AVGPOOL2D_LAYER_HPP
#define KL_AVGPOOL2D_LAYER_HPP

#include <cnn/layers/cache/layer_cache_key.hpp>
#include <cnn/layers/layer.hpp>
#include <cnn/options/pooling2d_options.hpp>

#include <core/shape.hpp>
#include <core/tensor.hpp>
#include <core/tensor_pool.hpp>

#include <cstddef>

namespace kl
{

    class AvgPool2dLayer final : public Layer
    {
    public:
        explicit AvgPool2dLayer(Pooling2dOptions options = {});

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

        const Pooling2dOptions &options() const;

    private:
        void prepare_cache(
            const Tensor &input);

        std::size_t output_size(
            std::size_t input_size,
            std::size_t kernel_size,
            std::size_t padding,
            std::size_t stride) const;

    private:
        Pooling2dOptions options_;

        LayerCacheKey cache_key_;
        Shape cached_output_shape_;

        bool last_forward_used_fast_path_ = false;

        Shape last_input_shape_;
        bool has_last_input_shape_ = false;
    };

}

#endif // KL_AVGPOOL2D_LAYER_HPP