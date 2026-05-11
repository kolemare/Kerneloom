#ifndef KL_TENSOR_POOL_HPP
#define KL_TENSOR_POOL_HPP

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/layout.hpp>
#include <core/shape.hpp>
#include <core/storage.hpp>
#include <core/tensor.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace kl
{

    class TensorPool
    {
    public:
        TensorPool() = default;
        ~TensorPool() = default;

        TensorPool(const TensorPool &) = delete;
        TensorPool &operator=(const TensorPool &) = delete;

        TensorPool(TensorPool &&) noexcept = default;
        TensorPool &operator=(TensorPool &&) noexcept = default;

        Tensor &request(
            const Shape &shape,
            DType dtype,
            Device device,
            Layout layout = Layout::Unknown,
            Storage storage = Storage::RowMajor);

        void reset();
        void clear();

        std::size_t tensor_count() const;
        std::size_t active_count() const;
        std::size_t inactive_count() const;

    private:
        struct TensorEntry
        {
            std::unique_ptr<Tensor> tensor;
            bool active = false;
        };

        TensorEntry *find_reusable(
            const Shape &shape,
            DType dtype,
            Device device,
            Layout layout,
            Storage storage);

        bool matches(
            const Tensor &tensor,
            const Shape &shape,
            DType dtype,
            Device device,
            Layout layout,
            Storage storage) const;

    private:
        std::vector<TensorEntry> tensors_;
    };

}

#endif // KL_TENSOR_POOL_HPP