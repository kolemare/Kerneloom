#ifndef KL_LINEAR_LAYER_HPP
#define KL_LINEAR_LAYER_HPP

#include <cnn/layers/layer.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>
#include <core/tensor_pool.hpp>

#include <cstddef>

namespace kl
{

    class LinearLayer final : public Layer
    {
    public:
        LinearLayer(
            std::size_t input_features,
            std::size_t output_features,
            DType dtype = DType::Float32,
            Device device = Device::cpu(),
            bool use_bias = true);

        bool verify() const override;

        Tensor &forward(
            Tensor &input,
            TensorPool &pool) override;

        Tensor &backward(
            Tensor &grad_output,
            TensorPool &pool) override;

        const Tensor &weights() const;
        const Tensor &bias() const;

        std::size_t input_features() const;
        std::size_t output_features() const;
        bool use_bias() const;

    private:
        std::size_t input_features_;
        std::size_t output_features_;

        DType dtype_;
        Device device_;
        bool use_bias_;

        Tensor weights_;
        Tensor bias_;

        const Tensor *last_input_ = nullptr;
    };

}

#endif // KL_LINEAR_LAYER_HPP