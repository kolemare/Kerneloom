#ifndef KL_MAXPOOL2D_LAYER_HPP
#define KL_MAXPOOL2D_LAYER_HPP

#include <cnn/layers/layer.hpp>
#include <cnn/options/pooling2d_options.hpp>

#include <core/shape.hpp>
#include <core/tensor.hpp>
#include <core/tensor_pool.hpp>

#include <cstddef>
#include <memory>

namespace kl
{

    class MaxPool2dLayer final : public Layer
    {
    public:
        explicit MaxPool2dLayer(Pooling2dOptions options = {});

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

        Tensor &indices();

        bool hasIndices() const;

        const Pooling2dOptions &options() const;

    private:
        void prepare_indices(
            const Shape &shape,
            Device device);

        std::size_t output_size(
            std::size_t input_size,
            std::size_t kernel_size,
            std::size_t padding,
            std::size_t stride) const;

    private:
        Pooling2dOptions options_;

        Shape last_input_shape_;
        bool has_last_input_shape_ = false;

        std::unique_ptr<Tensor> indices_;
    };

}

#endif // KL_MAXPOOL2D_LAYER_HPP