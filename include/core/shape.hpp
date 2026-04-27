#pragma once

#include <cstddef>

namespace kl
{

    class Shape2D
    {
    public:
        Shape2D();
        Shape2D(std::size_t rows, std::size_t cols);

        std::size_t rows() const;
        std::size_t cols() const;
        std::size_t size() const;

    private:
        std::size_t rows_;
        std::size_t cols_;
    };

}