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

    inline Tensor cublasLinearForwardFloat32(
        const Tensor &input,
        const Tensor &weights,
        std::size_t batch_size,
        std::size_t input_features,
        std::size_t output_features)
    {
        Tensor output(
            Shape{batch_size, output_features},
            DType::Float32,
            Device::cuda());

        CublasHandle handle;

        const float alpha = 1.0F;
        const float beta = 0.0F;

        const auto status = cublasSgemm(
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
            static_cast<int>(output_features));

        if (status != CUBLAS_STATUS_SUCCESS)
        {
            throw std::runtime_error("cuBLAS linear forward failed");
        }

        return output;
    }

}

#endif // KL_ENABLE_CUDA

#endif // KL_TEST_CUBLAS_LINEAR_HPP