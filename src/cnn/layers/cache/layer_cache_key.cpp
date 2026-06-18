#include <cnn/layers/cache/layer_cache_key.hpp>

#include <core/tensor.hpp>

namespace kl
{

    LayerCacheKey::LayerCacheKey() = default;

    LayerCacheKey::LayerCacheKey(
        const Tensor &tensor)
    {
        capture(
            tensor);
    }

    bool LayerCacheKey::valid() const
    {
        return valid_;
    }

    void LayerCacheKey::capture(
        const Tensor &tensor)
    {
        shape_ =
            tensor.shape();

        dtype_ =
            tensor.dtype();

        device_type_ =
            tensor.device().type();

        layout_ =
            tensor.layout();

        storage_ =
            tensor.storage();

        valid_ =
            true;
    }

    bool LayerCacheKey::matches(
        const Tensor &tensor) const
    {
        if (!valid_)
        {
            return false;
        }

        return shape_ == tensor.shape() &&
               dtype_ == tensor.dtype() &&
               device_type_ == tensor.device().type() &&
               layout_ == tensor.layout() &&
               storage_ == tensor.storage();
    }

    void LayerCacheKey::reset()
    {
        valid_ =
            false;

        shape_ =
            Shape{};

        dtype_ =
            DType::Float32;

        device_type_ =
            DeviceType::CPU;

        layout_ =
            Layout::Unknown;

        storage_ =
            Storage::RowMajor;
    }

    const Shape &LayerCacheKey::shape() const
    {
        return shape_;
    }

    DType LayerCacheKey::dtype() const
    {
        return dtype_;
    }

    DeviceType LayerCacheKey::device_type() const
    {
        return device_type_;
    }

    Layout LayerCacheKey::layout() const
    {
        return layout_;
    }

    Storage LayerCacheKey::storage() const
    {
        return storage_;
    }

}