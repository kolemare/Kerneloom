#ifndef KL_TEST_CUDNN_ACTIVATION_HPP
#define KL_TEST_CUDNN_ACTIVATION_HPP

#ifdef KL_ENABLE_CUDA

#include "vendor/cuda/cudnn_maxpool2d.cuh"

#include <core/tensor.hpp>

#include <cudnn.h>

#include <stdexcept>
#include <string>

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
                    std::string("Failed to create cuDNN activation descriptor: ") +
                    cudnnGetErrorString(create_status));
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
                    std::string("Failed to set cuDNN activation descriptor: ") +
                    cudnnGetErrorString(set_status));
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
        const CudnnTensorDescriptor &tensor_descriptor,
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
                tensor_descriptor.get(),
                input.data(),
                &beta,
                tensor_descriptor.get(),
                output.data());

        if (status != CUDNN_STATUS_SUCCESS)
        {
            throw std::runtime_error(
                std::string("cuDNN activation forward failed: ") +
                cudnnGetErrorString(status));
        }
    }

    inline void cudnnSoftmaxForwardFloat32(
        CudnnHandle &handle,
        const CudnnTensorDescriptor &tensor_descriptor,
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
                tensor_descriptor.get(),
                input.data(),
                &beta,
                tensor_descriptor.get(),
                output.data());

        if (status != CUDNN_STATUS_SUCCESS)
        {
            throw std::runtime_error(
                std::string("cuDNN softmax forward failed: ") +
                cudnnGetErrorString(status));
        }
    }

}

#endif // KL_ENABLE_CUDA

#endif // KL_TEST_CUDNN_ACTIVATION_HPP