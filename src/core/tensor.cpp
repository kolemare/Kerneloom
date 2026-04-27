#include <core/tensor.hpp>

#include <stdexcept>

namespace kl
{

    Tensor::Tensor()
        : shape_(0, 0), device_(Device::cpu()) {}

    Tensor::Tensor(Shape2D shape, Device device)
        : shape_(shape), device_(device), data_(shape.size(), 0.0f) {}

    Tensor::Tensor(Shape2D shape, std::vector<float> data, Device device)
        : shape_(shape), device_(device), data_(std::move(data))
    {
        if (data_.size() != shape_.size())
        {
            throw std::runtime_error("Tensor data size does not match shape");
        }
    }

    const Shape2D &Tensor::shape() const
    {
        return shape_;
    }

    const Device &Tensor::device() const
    {
        return device_;
    }

    std::size_t Tensor::rows() const
    {
        return shape_.rows();
    }

    std::size_t Tensor::cols() const
    {
        return shape_.cols();
    }

    std::size_t Tensor::size() const
    {
        return shape_.size();
    }

    float *Tensor::data()
    {
        return data_.data();
    }

    const float *Tensor::data() const
    {
        return data_.data();
    }

    std::vector<float> &Tensor::host_data()
    {
        return data_;
    }

    const std::vector<float> &Tensor::host_data() const
    {
        return data_;
    }

    float &Tensor::operator()(std::size_t row, std::size_t col)
    {
        return data_[row * cols() + col];
    }

    const float &Tensor::operator()(std::size_t row, std::size_t col) const
    {
        return data_[row * cols() + col];
    }

}