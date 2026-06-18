#ifndef KL_LAYER_CACHE_KEY_HPP
#define KL_LAYER_CACHE_KEY_HPP

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/layout.hpp>
#include <core/shape.hpp>
#include <core/storage.hpp>

namespace kl
{

    class Tensor;

    class LayerCacheKey
    {
    public:
        LayerCacheKey();

        explicit LayerCacheKey(
            const Tensor &tensor);

        bool valid() const;

        void capture(
            const Tensor &tensor);

        bool matches(
            const Tensor &tensor) const;

        void reset();

        const Shape &shape() const;
        DType dtype() const;
        DeviceType device_type() const;
        Layout layout() const;
        Storage storage() const;

    private:
        bool valid_ = false;

        Shape shape_;
        DType dtype_ = DType::Float32;
        DeviceType device_type_ = DeviceType::CPU;
        Layout layout_ = Layout::Unknown;
        Storage storage_ = Storage::RowMajor;
    };

}

#endif // KL_LAYER_CACHE_KEY_HPP