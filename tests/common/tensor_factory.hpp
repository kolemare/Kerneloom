#ifndef KL_TEST_TENSOR_FACTORY_HPP
#define KL_TEST_TENSOR_FACTORY_HPP

#include "common/dtype_dispatch.hpp"

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>

#include <cstddef>
#include <random>
#include <utility>

namespace kl::test
{

    inline Tensor makeTensor(
        Shape shape,
        DType dtype,
        Device device)
    {
        return Tensor(
            std::move(shape),
            dtype,
            device);
    }

    inline Tensor makeZeroTensor(
        Shape shape,
        DType dtype,
        Device device)
    {
        Tensor tensor(
            std::move(shape),
            dtype,
            Device::cpu());

        dispatchFloatDType(dtype, [&]<typename T>()
                           {
            T* data = static_cast<T*>(tensor.data());

            for (std::size_t i = 0; i < tensor.numel(); ++i)
            {
                data[i] = static_cast<T>(0);
            } });

        return tensor.to(device);
    }

    inline Tensor makeRandomTensor(
        Shape shape,
        DType dtype,
        Device device,
        double min_value = -1.0,
        double max_value = 1.0,
        unsigned int seed = 42)
    {
        Tensor tensor(
            std::move(shape),
            dtype,
            Device::cpu());

        std::mt19937 generator(seed);
        std::uniform_real_distribution<double> distribution(
            min_value,
            max_value);

        dispatchFloatDType(dtype, [&]<typename T>()
                           {
            T* data = static_cast<T*>(tensor.data());

            for (std::size_t i = 0; i < tensor.numel(); ++i)
            {
                data[i] = static_cast<T>(distribution(generator));
            } });

        return tensor.to(device);
    }

}

#endif // KL_TEST_TENSOR_FACTORY_HPP