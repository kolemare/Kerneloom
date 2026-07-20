#ifndef KL_TEST_CUBLAS_LINEAR_HPP
#define KL_TEST_CUBLAS_LINEAR_HPP

#ifdef KL_ENABLE_CUDA

#include "vendor/cuda/cublas_handle.cuh"

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>

#include <cublas_v2.h>

#include <cstddef>
#include <stdexcept>

namespace kl::test
{

    inline void check_cublas(
        cublasStatus_t status,
        const char *message)
    {
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            throw std::runtime_error(
                message);
        }
    }

    inline Tensor makeOnesCudaFloat32(
        std::size_t size)
    {
        Tensor ones_cpu(
            Shape{size},
            DType::Float32,
            Device::cpu());

        float *data =
            static_cast<float *>(ones_cpu.data());

        for (std::size_t i = 0; i < size; ++i)
        {
            data[i] =
                1.0F;
        }

        return ones_cpu.to(
            Device::cuda());
    }

    inline void cublasLinearForwardFloat32(
        CublasHandle &handle,
        const Tensor &input,
        const Tensor &weights,
        Tensor &output)
    {
        const std::size_t batch_size =
            input.shape()[0];

        const std::size_t input_features =
            input.shape()[1];

        const std::size_t output_features =
            weights.shape()[0];

        const float alpha =
            1.0F;

        const float beta =
            0.0F;

        check_cublas(
            cublasSgemm(
                handle.get(),
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                static_cast<int>(output_features),
                static_cast<int>(batch_size),
                static_cast<int>(input_features),
                &alpha,
                static_cast<const float *>(weights.data()),
                static_cast<int>(input_features),
                static_cast<const float *>(input.data()),
                static_cast<int>(input_features),
                &beta,
                static_cast<float *>(output.data()),
                static_cast<int>(output_features)),
            "cuBLAS linear forward failed");
    }

    inline void cublasLinearForwardFloat32(
        CublasHandle &handle,
        const Tensor &input,
        const Tensor &weights,
        const Tensor &bias,
        const Tensor &ones,
        Tensor &output)
    {
        cublasLinearForwardFloat32(
            handle,
            input,
            weights,
            output);

        const std::size_t batch_size =
            input.shape()[0];

        const std::size_t output_features =
            weights.shape()[0];

        const float alpha =
            1.0F;

        check_cublas(
            cublasSger(
                handle.get(),
                static_cast<int>(output_features),
                static_cast<int>(batch_size),
                &alpha,
                static_cast<const float *>(bias.data()),
                1,
                static_cast<const float *>(ones.data()),
                1,
                static_cast<float *>(output.data()),
                static_cast<int>(output_features)),
            "cuBLAS linear bias add failed");
    }

    inline Tensor cublasLinearForwardFloat32(
        const Tensor &input,
        const Tensor &weights)
    {
        Tensor output(
            Shape{input.shape()[0], weights.shape()[0]},
            DType::Float32,
            Device::cuda());

        CublasHandle handle;

        cublasLinearForwardFloat32(
            handle,
            input,
            weights,
            output);

        return output;
    }

    inline Tensor cublasLinearForwardFloat32(
        const Tensor &input,
        const Tensor &weights,
        const Tensor &bias)
    {
        Tensor output(
            Shape{input.shape()[0], weights.shape()[0]},
            DType::Float32,
            Device::cuda());

        Tensor ones =
            makeOnesCudaFloat32(
                input.shape()[0]);

        CublasHandle handle;

        cublasLinearForwardFloat32(
            handle,
            input,
            weights,
            bias,
            ones,
            output);

        return output;
    }

}

#endif // KL_ENABLE_CUDA

#endif // KL_TEST_CUBLAS_LINEAR_HPP