#ifndef KL_TEST_CUBLAS_LINEAR_BACKWARD_HPP
#define KL_TEST_CUBLAS_LINEAR_BACKWARD_HPP

#ifdef KL_ENABLE_CUDA

#include "vendor/cuda/cublas_linear.cuh"

#include <core/tensor.hpp>

#include <cublas_v2.h>

#include <stdexcept>

namespace kl::test
{

    inline void check_cublas_linear_backward(
        cublasStatus_t status,
        const char *message)
    {
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            throw std::runtime_error(
                message);
        }
    }

    inline void cublasLinearBackwardGradInputFloat32(
        CublasHandle &handle,
        const Tensor &weights,
        const Tensor &grad_output,
        Tensor &grad_input)
    {
        const int batch_size =
            static_cast<int>(grad_output.shape()[0]);

        const int output_features =
            static_cast<int>(grad_output.shape()[1]);

        const int input_features =
            static_cast<int>(weights.shape()[1]);

        const float alpha =
            1.0F;

        const float beta =
            0.0F;

        check_cublas_linear_backward(
            cublasSgemm(
                handle.get(),
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                input_features,
                batch_size,
                output_features,
                &alpha,
                static_cast<const float *>(weights.data()),
                input_features,
                static_cast<const float *>(grad_output.data()),
                output_features,
                &beta,
                static_cast<float *>(grad_input.data()),
                input_features),
            "cuBLAS linear backward grad_input failed");
    }

    inline Tensor cublasLinearBackwardGradInputFloat32(
        const Tensor &weights,
        const Tensor &grad_output)
    {
        Tensor grad_input(
            Shape{
                grad_output.shape()[0],
                weights.shape()[1]},
            grad_output.dtype(),
            grad_output.device());

        CublasHandle handle;

        cublasLinearBackwardGradInputFloat32(
            handle,
            weights,
            grad_output,
            grad_input);

        return grad_input;
    }

    inline void cublasLinearBackwardGradWeightsFloat32(
        CublasHandle &handle,
        const Tensor &input,
        const Tensor &grad_output,
        Tensor &grad_weights)
    {
        const int batch_size =
            static_cast<int>(input.shape()[0]);

        const int input_features =
            static_cast<int>(input.shape()[1]);

        const int output_features =
            static_cast<int>(grad_output.shape()[1]);

        const float alpha =
            1.0F;

        const float beta =
            0.0F;

        check_cublas_linear_backward(
            cublasSgemm(
                handle.get(),
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                input_features,
                output_features,
                batch_size,
                &alpha,
                static_cast<const float *>(input.data()),
                input_features,
                static_cast<const float *>(grad_output.data()),
                output_features,
                &beta,
                static_cast<float *>(grad_weights.data()),
                input_features),
            "cuBLAS linear backward grad_weights failed");
    }

    inline Tensor cublasLinearBackwardGradWeightsFloat32(
        const Tensor &input,
        const Tensor &grad_output)
    {
        Tensor grad_weights(
            Shape{
                grad_output.shape()[1],
                input.shape()[1]},
            grad_output.dtype(),
            grad_output.device());

        CublasHandle handle;

        cublasLinearBackwardGradWeightsFloat32(
            handle,
            input,
            grad_output,
            grad_weights);

        return grad_weights;
    }

    inline void cublasLinearBackwardGradBiasFloat32(
        CublasHandle &handle,
        const Tensor &grad_output,
        const Tensor &ones,
        Tensor &grad_bias)
    {
        const int batch_size =
            static_cast<int>(grad_output.shape()[0]);

        const int output_features =
            static_cast<int>(grad_output.shape()[1]);

        const float alpha =
            1.0F;

        const float beta =
            0.0F;

        check_cublas_linear_backward(
            cublasSgemv(
                handle.get(),
                CUBLAS_OP_N,
                output_features,
                batch_size,
                &alpha,
                static_cast<const float *>(grad_output.data()),
                output_features,
                static_cast<const float *>(ones.data()),
                1,
                &beta,
                static_cast<float *>(grad_bias.data()),
                1),
            "cuBLAS linear backward grad_bias failed");
    }

    inline Tensor cublasLinearBackwardGradBiasFloat32(
        const Tensor &grad_output)
    {
        Tensor grad_bias(
            Shape{
                grad_output.shape()[1]},
            grad_output.dtype(),
            grad_output.device());

        const Tensor ones =
            makeOnesCudaFloat32(
                grad_output.shape()[0]);

        CublasHandle handle;

        cublasLinearBackwardGradBiasFloat32(
            handle,
            grad_output,
            ones,
            grad_bias);

        return grad_bias;
    }

}

#endif // KL_ENABLE_CUDA

#endif // KL_TEST_CUBLAS_LINEAR_BACKWARD_HPP