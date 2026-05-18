#include <cnn/layers/linear_layer.hpp>

#include <core/layout.hpp>
#include <core/storage.hpp>

#include <ops/linear.hpp>

#include <stdexcept>

namespace kl
{

    LinearLayer::LinearLayer(
        std::size_t input_features,
        std::size_t output_features,
        DType dtype,
        Device device,
        bool use_bias)
        : input_features_(input_features),
          output_features_(output_features),
          dtype_(dtype),
          device_(device),
          use_bias_(use_bias),
          weights_(
              Shape{output_features, input_features},
              dtype,
              device,
              Layout::Unknown,
              Storage::RowMajor)
    {
        if (use_bias_)
        {
            bias_ = std::make_unique<Tensor>(
                Shape{output_features},
                dtype,
                device,
                Layout::Unknown,
                Storage::RowMajor);
        }
    }

    void LinearLayer::initializeBiases(const InitializerType &type)
    {
        if (bias_ != nullptr)
        {
            Initializer::initialize(*bias_, type);
        }
    }

    void LinearLayer::initializeWeights(const InitializerType &type)
    {
        Initializer::initialize(weights_, type);
    }

    void LinearLayer::prepareTraining()
    {
        if (grad_weights_ == nullptr)
        {
            grad_weights_ = std::make_unique<Tensor>(
                weights_.shape(),
                weights_.dtype(),
                weights_.device(),
                weights_.layout(),
                weights_.storage());
        }

        if (bias_ != nullptr && grad_bias_ == nullptr)
        {
            grad_bias_ = std::make_unique<Tensor>(
                bias_->shape(),
                bias_->dtype(),
                bias_->device(),
                bias_->layout(),
                bias_->storage());
        }
    }

    bool LinearLayer::verify() const
    {
        if (input_features_ == 0 || output_features_ == 0)
        {
            return false;
        }

        if (dtype_ != DType::Float32)
        {
            return false;
        }

        if (weights_.rank() != 2 ||
            weights_.shape()[0] != output_features_ ||
            weights_.shape()[1] != input_features_)
        {
            return false;
        }

        if (weights_.dtype() != dtype_ ||
            weights_.device().type() != device_.type() ||
            weights_.storage() != Storage::RowMajor)
        {
            return false;
        }

        if (use_bias_)
        {
            if (bias_ == nullptr)
            {
                return false;
            }

            if (bias_->rank() != 1 ||
                bias_->shape()[0] != output_features_ ||
                bias_->dtype() != dtype_ ||
                bias_->device().type() != device_.type() ||
                bias_->storage() != Storage::RowMajor)
            {
                return false;
            }
        }

        if (!use_bias_ && bias_ != nullptr)
        {
            return false;
        }

        return true;
    }

    Shape LinearLayer::output_shape(
        const Shape &input_shape) const
    {
        return Shape{
            input_shape[0],
            output_features_};
    }

    Tensor &LinearLayer::forward(
        Tensor &input,
        TensorPool &pool)
    {
        Tensor &result = pool.request(
            output_shape(input.shape()),
            dtype_,
            device_,
            Layout::Unknown,
            Storage::RowMajor);

        const Tensor *bias = nullptr;

        if (bias_ != nullptr)
        {
            bias = bias_.get();
        }

        linear(
            input,
            weights_,
            bias,
            result);

        last_input_ = &input;

        return result;
    }

    Tensor &LinearLayer::backward(
        Tensor &grad_output,
        TensorPool &pool)
    {
        if (last_input_ == nullptr)
        {
            throw std::runtime_error("LinearLayer::backward called before forward");
        }

        Tensor &grad_input = pool.request(
            last_input_->shape(),
            grad_output.dtype(),
            grad_output.device(),
            grad_output.layout(),
            grad_output.storage());

        return grad_input;
    }

    Tensor &LinearLayer::weights()
    {
        return weights_;
    }

    Tensor &LinearLayer::bias()
    {
        if (bias_ == nullptr)
        {
            throw std::runtime_error("LinearLayer::bias requested but bias is disabled");
        }

        return *bias_;
    }

    Tensor &LinearLayer::gradWeights()
    {
        if (grad_weights_ == nullptr)
        {
            throw std::runtime_error("LinearLayer::gradWeights called before prepareTraining");
        }

        return *grad_weights_;
    }

    Tensor &LinearLayer::gradBias()
    {
        if (bias_ == nullptr)
        {
            throw std::runtime_error("LinearLayer::gradBias requested but bias is disabled");
        }

        if (grad_bias_ == nullptr)
        {
            throw std::runtime_error("LinearLayer::gradBias called before prepareTraining");
        }

        return *grad_bias_;
    }

    std::size_t LinearLayer::input_features() const
    {
        return input_features_;
    }

    std::size_t LinearLayer::output_features() const
    {
        return output_features_;
    }

    bool LinearLayer::use_bias() const
    {
        return use_bias_;
    }

    bool LinearLayer::hasBias() const
    {
        return bias_ != nullptr;
    }
}