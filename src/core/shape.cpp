#include <core/shape.hpp>

#include <numeric>
#include <stdexcept>
#include <utility>

namespace kl
{

    Shape::Shape() = default;

    Shape::Shape(std::initializer_list<std::size_t> dims)
        : dims_(dims)
    {
    }

    Shape::Shape(std::vector<std::size_t> dims)
        : dims_(std::move(dims))
    {
    }

    std::size_t Shape::rank() const
    {
        return dims_.size();
    }

    std::size_t Shape::numel() const
    {
        if (dims_.empty())
        {
            return 1; // scalar
        }

        return std::accumulate(
            dims_.begin(),
            dims_.end(),
            static_cast<std::size_t>(1),
            [](std::size_t a, std::size_t b)
            {
                return a * b;
            });
    }

    bool Shape::empty() const
    {
        return dims_.empty();
    }

    std::size_t Shape::operator[](std::size_t index) const
    {
        if (index >= dims_.size())
        {
            throw std::out_of_range("Shape dimension index out of range");
        }

        return dims_[index];
    }

    bool Shape::operator==(const Shape &other) const
    {
        return dims_ == other.dims_;
    }

    bool Shape::operator!=(const Shape &other) const
    {
        return !(*this == other);
    }

    const std::vector<std::size_t> &Shape::dims() const
    {
        return dims_;
    }

}