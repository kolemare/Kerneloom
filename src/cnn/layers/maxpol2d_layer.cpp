#include <cnn/layers/maxpool2d_layer.hpp>

#include <core/layout.hpp>
#include <core/storage.hpp>

#include <ops/backward_maxpool2d.hpp>
#include <ops/maxpool2d.hpp>
#include <ops/maxpool2d_with_indices.hpp>

#include <stdexcept>

namespace kl
{

    MaxPool2dLayer::MaxPool2dLayer(Pooling2dOptions options)
        : options_(options)
    {
    }

    void MaxPool2dLayer::initializeBiases(const InitializerType &type)
    {
        (void)type;
    }

    void MaxPool2dLayer::initializeWeights(const InitializerType &type)
    {
        (void)type;
    }

    void MaxPool2dLayer::prepareTraining()
    {
        mode_ = LayerMode::Training;
    }

    bool MaxPool2dLayer::verify() const
    {
        if (options_.kernel_h == 0 || options_.kernel_w == 0)
        {
            return false;
        }

        if (options_.stride_h == 0 || options_.stride_w == 0)
        {
            return false;
        }

        return true;
    }

    Shape MaxPool2dLayer::output_shape(
        const Shape &input_shape) const
    {
        const std::size_t output_height = output_size(
            input_shape[2],
            options_.kernel_h,
            options_.padding_h,
            options_.stride_h);

        const std::size_t output_width = output_size(
            input_shape[3],
            options_.kernel_w,
            options_.padding_w,
            options_.stride_w);

        return Shape{
            input_shape[0],
            input_shape[1],
            output_height,
            output_width};
    }

    Tensor &MaxPool2dLayer::forward(
        Tensor &input,
        TensorPool &pool)
    {
        const bool cache_hit =
            cache_key_.matches(
                input) &&
            cached_mode_ == mode();

        prepare_cache(
            input);

        Tensor &result = pool.request(
            cached_output_shape_,
            input.dtype(),
            input.device(),
            Layout::NCHW,
            Storage::RowMajor);

        if (mode() == LayerMode::Training)
        {
            prepare_indices(
                cached_output_shape_,
                input.device());

            if (cache_hit)
            {
                maxpool2d_with_indices_unchecked(
                    input,
                    result,
                    *indices_,
                    options_);
            }
            else
            {
                maxpool2d_with_indices(
                    input,
                    result,
                    *indices_,
                    options_);
            }
        }
        else
        {
            if (cache_hit)
            {
                maxpool2d_unchecked(
                    input,
                    result,
                    options_);
            }
            else
            {
                maxpool2d(
                    input,
                    result,
                    options_);
            }
        }

        cached_mode_ =
            mode();

        last_forward_used_fast_path_ =
            cache_hit;

        last_input_shape_ =
            cache_key_.shape();

        has_last_input_shape_ =
            true;

        return result;
    }

    Tensor &MaxPool2dLayer::backward(
        Tensor &grad_output,
        TensorPool &pool)
    {
        if (!has_last_input_shape_)
        {
            throw std::runtime_error("MaxPool2dLayer::backward called before forward");
        }

        if (mode() != LayerMode::Training)
        {
            throw std::runtime_error("MaxPool2dLayer::backward called while layer is not in training mode");
        }

        if (indices_ == nullptr)
        {
            throw std::runtime_error("MaxPool2dLayer::backward called without saved indices");
        }

        Tensor &grad_input = pool.request(
            last_input_shape_,
            grad_output.dtype(),
            grad_output.device(),
            Layout::NCHW,
            Storage::RowMajor);

        if (last_forward_used_fast_path_)
        {
            backward_maxpool2d_unchecked(
                *indices_,
                grad_output,
                grad_input);
        }
        else
        {
            backward_maxpool2d(
                *indices_,
                grad_output,
                grad_input);
        }

        return grad_input;
    }

    Tensor &MaxPool2dLayer::indices()
    {
        if (indices_ == nullptr)
        {
            throw std::runtime_error("MaxPool2dLayer::indices called before training forward");
        }

        return *indices_;
    }

    bool MaxPool2dLayer::hasIndices() const
    {
        return indices_ != nullptr;
    }

    const Pooling2dOptions &MaxPool2dLayer::options() const
    {
        return options_;
    }

    void MaxPool2dLayer::prepare_cache(
        const Tensor &input)
    {
        if (cache_key_.matches(input))
        {
            return;
        }

        cached_output_shape_ =
            output_shape(
                input.shape());

        cache_key_.capture(
            input);
    }

    void MaxPool2dLayer::prepare_indices(
        const Shape &shape,
        Device device)
    {
        if (indices_ != nullptr && indices_->shape() == shape)
        {
            return;
        }

        indices_ = std::make_unique<Tensor>(
            shape,
            DType::Int32,
            device,
            Layout::NCHW,
            Storage::RowMajor);
    }

    std::size_t MaxPool2dLayer::output_size(
        std::size_t input_size,
        std::size_t kernel_size,
        std::size_t padding,
        std::size_t stride) const
    {
        return (input_size + 2 * padding - kernel_size) / stride + 1;
    }

}