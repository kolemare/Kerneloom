#ifndef KL_TEST_CUDNN_ACTIVATION_BACKWARD_HPP
#define KL_TEST_CUDNN_ACTIVATION_BACKWARD_HPP

#ifdef KL_ENABLE_CUDA

#include "vendor/cuda/cudnn_activation.cuh"

#include <core/tensor.hpp>

#include <cudnn.h>

#include <stdexcept>

namespace kl::test
{

    inline void cudnnActivationBackwardFloat32(
        CudnnHandle &handle,
        const CudnnTensorDescriptor &activation_output_descriptor,
        const CudnnTensorDescriptor &gradient_descriptor,
        const CudnnTensorDescriptor &input_descriptor,
        const CudnnTensorDescriptor &output_gradient_descriptor,
        const CudnnActivationDescriptor &activation_descriptor,
        const Tensor &activation_output,
        const Tensor &gradient,
        const Tensor &input,
        Tensor &output_gradient)
    {
        const float alpha =
            1.0F;

        const float beta =
            0.0F;

        const cudnnStatus_t status =
            cudnnActivationBackward(
                handle.get(),
                activation_descriptor.get(),
                &alpha,
                activation_output_descriptor.get(),
                activation_output.data(),
                gradient_descriptor.get(),
                gradient.data(),
                input_descriptor.get(),
                input.data(),
                &beta,
                output_gradient_descriptor.get(),
                output_gradient.data());

        if (status != CUDNN_STATUS_SUCCESS)
        {
            throw std::runtime_error(
                "cuDNN activation backward failed");
        }
    }

    inline void cudnnActivationBackwardFloat32(
        CudnnHandle &handle,
        const CudnnTensorDescriptor &tensor_descriptor,
        const CudnnActivationDescriptor &activation_descriptor,
        const Tensor &activation_output,
        const Tensor &gradient,
        const Tensor &input,
        Tensor &output_gradient)
    {
        cudnnActivationBackwardFloat32(
            handle,
            tensor_descriptor,
            tensor_descriptor,
            tensor_descriptor,
            tensor_descriptor,
            activation_descriptor,
            activation_output,
            gradient,
            input,
            output_gradient);
    }

    inline Tensor cudnnActivationBackwardFloat32(
        const Tensor &input,
        const Tensor &activation_output,
        const Tensor &gradient,
        cudnnActivationMode_t mode)
    {
        Tensor output_gradient(
            gradient.shape(),
            gradient.dtype(),
            gradient.device());

        CudnnHandle handle;

        CudnnTensorDescriptor input_descriptor(
            input);

        CudnnTensorDescriptor activation_output_descriptor(
            activation_output);

        CudnnTensorDescriptor gradient_descriptor(
            gradient);

        CudnnTensorDescriptor output_gradient_descriptor(
            output_gradient);

        CudnnActivationDescriptor activation_descriptor(
            mode);

        cudnnActivationBackwardFloat32(
            handle,
            activation_output_descriptor,
            gradient_descriptor,
            input_descriptor,
            output_gradient_descriptor,
            activation_descriptor,
            activation_output,
            gradient,
            input,
            output_gradient);

        return output_gradient;
    }

    inline void cudnnSoftmaxBackwardFloat32(
        CudnnHandle &handle,
        const CudnnTensorDescriptor &activation_output_descriptor,
        const CudnnTensorDescriptor &gradient_descriptor,
        const CudnnTensorDescriptor &output_gradient_descriptor,
        const Tensor &activation_output,
        const Tensor &gradient,
        Tensor &output_gradient)
    {
        const float alpha =
            1.0F;

        const float beta =
            0.0F;

        const cudnnStatus_t status =
            cudnnSoftmaxBackward(
                handle.get(),
                CUDNN_SOFTMAX_ACCURATE,
                CUDNN_SOFTMAX_MODE_CHANNEL,
                &alpha,
                activation_output_descriptor.get(),
                activation_output.data(),
                gradient_descriptor.get(),
                gradient.data(),
                &beta,
                output_gradient_descriptor.get(),
                output_gradient.data());

        if (status != CUDNN_STATUS_SUCCESS)
        {
            throw std::runtime_error(
                "cuDNN softmax backward failed");
        }
    }

    inline void cudnnSoftmaxBackwardFloat32(
        CudnnHandle &handle,
        const CudnnTensorDescriptor &tensor_descriptor,
        const Tensor &activation_output,
        const Tensor &gradient,
        Tensor &output_gradient)
    {
        cudnnSoftmaxBackwardFloat32(
            handle,
            tensor_descriptor,
            tensor_descriptor,
            tensor_descriptor,
            activation_output,
            gradient,
            output_gradient);
    }

    inline Tensor cudnnSoftmaxBackwardFloat32(
        const Tensor &activation_output,
        const Tensor &gradient)
    {
        Tensor output_gradient(
            gradient.shape(),
            gradient.dtype(),
            gradient.device());

        CudnnHandle handle;

        CudnnTensorDescriptor activation_output_descriptor(
            activation_output);

        CudnnTensorDescriptor gradient_descriptor(
            gradient);

        CudnnTensorDescriptor output_gradient_descriptor(
            output_gradient);

        cudnnSoftmaxBackwardFloat32(
            handle,
            activation_output_descriptor,
            gradient_descriptor,
            output_gradient_descriptor,
            activation_output,
            gradient,
            output_gradient);

        return output_gradient;
    }

}

#endif // KL_ENABLE_CUDA

#endif // KL_TEST_CUDNN_ACTIVATION_BACKWARD_HPP