#ifndef KL_SHAPE_HPP
#define KL_SHAPE_HPP

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace kl
{

    class Shape
    {
    public:
        Shape();
        Shape(std::initializer_list<std::size_t> dims);
        explicit Shape(std::vector<std::size_t> dims);

        std::size_t rank() const;
        std::size_t numel() const;

        bool empty() const;

        std::size_t operator[](std::size_t index) const;

        bool operator==(const Shape &other) const;
        bool operator!=(const Shape &other) const;

        const std::vector<std::size_t> &dims() const;

    private:
        std::vector<std::size_t> dims_;
    };

}

#endif // KL_SHAPE_HPP