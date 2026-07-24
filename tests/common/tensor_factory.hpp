#ifndef KL_TEST_TENSOR_FACTORY_HPP
#define KL_TEST_TENSOR_FACTORY_HPP

#include "common/dtype_dispatch.hpp"

#ifdef KL_ENABLE_CUDA
#include "vendor/cuda/curand_tensor_factory.cuh"
#endif

#ifdef KL_ENABLE_ROCM
#include "vendor/rocm/hiprand_tensor_factory.hiph"
#endif

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>

#include <cstddef>
#include <random>
#include <stdexcept>
#include <utility>

namespace kl::test
{

    namespace
    {

        inline Tensor makeRandomTensorCpu(
            Shape shape,
            DType dtype,
            double min_value,
            double max_value,
            unsigned int seed)
        {
            Tensor tensor(
                std::move(shape),
                dtype,
                Device::cpu());

            std::mt19937 generator(
                seed);

            std::uniform_real_distribution<double> distribution(
                min_value,
                max_value);

            dispatchFloatDType(dtype, [&]<typename T>()
                               {
                T *data =
                    static_cast<T *>(tensor.data());

                for (std::size_t i = 0;
                     i < tensor.numel();
                     ++i)
                {
                    data[i] =
                        static_cast<T>(
                            distribution(
                                generator));
                } });

            return tensor;
        }

    }

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
            T *data =
                static_cast<T *>(tensor.data());

            for (std::size_t i = 0;
                 i < tensor.numel();
                 ++i)
            {
                data[i] =
                    static_cast<T>(0);
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
#ifdef KL_ENABLE_CUDA
        if (device.type() == DeviceType::CUDA &&
            dtype == DType::Float32)
        {
            return makeRandomCudaTensorFloat32(
                std::move(shape),
                min_value,
                max_value,
                seed);
        }
#endif

#ifdef KL_ENABLE_ROCM
        if (device.type() == DeviceType::ROCM &&
            dtype == DType::Float32)
        {
            return makeRandomRocmTensorFloat32(
                std::move(shape),
                min_value,
                max_value,
                seed);
        }
#endif

        Tensor tensor =
            makeRandomTensorCpu(
                std::move(shape),
                dtype,
                min_value,
                max_value,
                seed);

        return tensor.to(device);
    }

    inline Tensor makeTransposedTensor2d(
        const Tensor &source,
        Device device)
    {
        if (source.shape().rank() != 2)
        {
            throw std::runtime_error(
                "makeTransposedTensor2d expects a rank-2 tensor");
        }

        Tensor source_cpu =
            source.to(Device::cpu());

        const std::size_t rows =
            source_cpu.shape()[0];

        const std::size_t columns =
            source_cpu.shape()[1];

        Tensor transposed_cpu(
            Shape{columns, rows},
            source_cpu.dtype(),
            Device::cpu());

        dispatchFloatDType(source_cpu.dtype(), [&]<typename T>()
                           {
            const T *source_data =
                static_cast<const T *>(source_cpu.data());

            T *transposed_data =
                static_cast<T *>(transposed_cpu.data());

            for (std::size_t row = 0;
                 row < rows;
                 ++row)
            {
                for (std::size_t column = 0;
                     column < columns;
                     ++column)
                {
                    transposed_data[column * rows + row] =
                        source_data[row * columns + column];
                }
            } });

        return transposed_cpu.to(device);
    }

}

#endif // KL_TEST_TENSOR_FACTORY_HPP