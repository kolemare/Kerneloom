#pragma once

#include <core/device.hpp>
#include <core/shape.hpp>

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace kl
{

    class Tensor
    {
    public:
        Tensor();
        Tensor(Shape2D shape, Device device = Device::cpu());
        Tensor(Shape2D shape, std::vector<float> data, Device device = Device::cpu());

        const Shape2D &shape() const;
        const Device &device() const;

        std::size_t rows() const;
        std::size_t cols() const;
        std::size_t size() const;

        float *data();
        const float *data() const;

        std::vector<float> &host_data();
        const std::vector<float> &host_data() const;

        float &operator()(std::size_t row, std::size_t col);
        const float &operator()(std::size_t row, std::size_t col) const;

    private:
        Shape2D shape_;
        Device device_;
        std::vector<float> data_;
    };

}