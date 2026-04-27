#include <core/shape.hpp>

namespace kl
{

    Shape2D::Shape2D()
        : rows_(0), cols_(0) {}

    Shape2D::Shape2D(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols) {}

    std::size_t Shape2D::rows() const
    {
        return rows_;
    }

    std::size_t Shape2D::cols() const
    {
        return cols_;
    }

    std::size_t Shape2D::size() const
    {
        return rows_ * cols_;
    }

}