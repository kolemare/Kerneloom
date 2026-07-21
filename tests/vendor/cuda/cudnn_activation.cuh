#ifndef KL_TEST_CUDNN_ACTIVATION_HPP
#define KL_TEST_CUDNN_ACTIVATION_HPP

#ifdef KL_ENABLE_CUDA

#include "vendor/cuda/cudnn_maxpool2d.cuh"

#include <core/tensor.hpp>

#include <cudnn.h>

#include <stdexcept>

namespace kl::test
{

    class CudnnActivationDescriptor
    {
    public:
        explicit CudnnActivationDescriptor(
            cudnnActivationMode_t mode)
        {
            const cudnnStatus_t create_status =
                cudnnCreateActivationDescriptor(
                    &descriptor_);

            if (create_status != CUDNN_STATUS_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to create cuDNN activation descriptor");
            }

            const cudnnStatus_t set_status =
                cudnnSetActivationDescriptor(
                    descriptor_,
                    mode,
                    CUDNN_PROPAGATE_NAN,
                    0.0);

            if (set_status != CUDNN_STATUS_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to set cuDNN activation descriptor");
            }
        }

        ~CudnnActivationDescriptor()
        {
            if (descriptor_ != nullptr)
            {
                (void)cudnnDestroyActivationDescriptor(
                    descriptor_);
            }
        }

        CudnnActivationDescriptor(
            const CudnnActivationDescriptor &) = delete;

        CudnnActivationDescriptor &operator=(
            const CudnnActivationDescriptor &) = delete;

        CudnnActivationDescriptor(
            CudnnActivationDescriptor &&) = delete;

        CudnnActivationDescriptor &operator=(
            CudnnActivationDescriptor &&) = delete;

        [[nodiscard]] cudnnActivationDescriptor_t get() const
        {
            return descriptor_;
        }

    private:
        cudnnActivationDescriptor_t descriptor_ =
            nullptr;
    };

    inline void cudnnActivationForwardFloat32(
        CudnnHandle &handle,
        const CudnnTensorDescriptor &input_descriptor,
        const CudnnTensorDescriptor &output_descriptor,
        const CudnnActivationDescriptor &activation_descriptor,
        const Tensor &input,
        Tensor &output)
    {
        const float alpha =
            1.0F;

        const float beta =
            0.0F;

        const cudnnStatus_t status =
            cudnnActivationForward(
                handle.get(),
                activation_descriptor.get(),
                &alpha,
                input_descriptor.get(),
                input.data(),
                &beta,
                output_descriptor.get(),
                output.data());

        if (status != CUDNN_STATUS_SUCCESS)
        {
            throw std::runtime_error(
                "cuDNN activation forward failed");
        }
    }

    inline void cudnnActivationForwardFloat32(
        CudnnHandle &handle,
        const CudnnTensorDescriptor &tensor_descriptor,
        const CudnnActivationDescriptor &activation_descriptor,
        const Tensor &input,
        Tensor &output)
    {
        cudnnActivationForwardFloat32(
            handle,
            tensor_descriptor,
            tensor_descriptor,
            activation_descriptor,
            input,
            output);
    }

    inline Tensor cudnnActivationForwardFloat32(
        const Tensor &input,
        cudnnActivationMode_t mode)
    {
        Tensor output(
            input.shape(),
            input.dtype(),
            input.device());

        CudnnHandle handle;

        CudnnTensorDescriptor input_descriptor(
            input);

        CudnnTensorDescriptor output_descriptor(
            output);

        CudnnActivationDescriptor activation_descriptor(
            mode);

        cudnnActivationForwardFloat32(
            handle,
            input_descriptor,
            output_descriptor,
            activation_descriptor,
            input,
            output);

        return output;
    }

    inline void cudnnSoftmaxForwardFloat32(
        CudnnHandle &handle,
        const CudnnTensorDescriptor &input_descriptor,
        const CudnnTensorDescriptor &output_descriptor,
        const Tensor &input,
        Tensor &output)
    {
        const float alpha =
            1.0F;

        const float beta =
            0.0F;

        const cudnnStatus_t status =
            cudnnSoftmaxForward(
                handle.get(),
                CUDNN_SOFTMAX_ACCURATE,
                CUDNN_SOFTMAX_MODE_CHANNEL,
                &alpha,
                input_descriptor.get(),
                input.data(),
                &beta,
                output_descriptor.get(),
                output.data());

        if (status != CUDNN_STATUS_SUCCESS)
        {
            throw std::runtime_error(
                "cuDNN softmax forward failed");
        }
    }

    inline void cudnnSoftmaxForwardFloat32(
        CudnnHandle &handle,
        const CudnnTensorDescriptor &tensor_descriptor,
        const Tensor &input,
        Tensor &output)
    {
        cudnnSoftmaxForwardFloat32(
            handle,
            tensor_descriptor,
            tensor_descriptor,
            input,
            output);
    }

    inline Tensor cudnnSoftmaxForwardFloat32(
        const Tensor &input)
    {
        Tensor output(
            input.shape(),
            input.dtype(),
            input.device());

        CudnnHandle handle;

        CudnnTensorDescriptor input_descriptor(
            input);

        CudnnTensorDescriptor output_descriptor(
            output);

        cudnnSoftmaxForwardFloat32(
            handle,
            input_descriptor,
            output_descriptor,
            input,
            output);

        return output;
    }

}

#endif // KL_ENABLE_CUDA

#endif // KL_TEST_CUDNN_ACTIVATION_HPP