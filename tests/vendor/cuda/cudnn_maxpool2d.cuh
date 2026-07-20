#ifndef KL_TEST_CUDNN_MAXPOOL2D_HPP
#define KL_TEST_CUDNN_MAXPOOL2D_HPP

#ifdef KL_ENABLE_CUDA

#include "vendor/cuda/cudnn_handle.cuh"

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>

#include <cudnn.h>

#include <stdexcept>

namespace kl::test
{

    class CudnnTensorDescriptor
    {
    public:
        CudnnTensorDescriptor(
            const Tensor &tensor)
        {
            if (tensor.shape().rank() != 4)
            {
                throw std::runtime_error(
                    "cuDNN tensor descriptor expects rank-4 tensor");
            }

            if (tensor.dtype() != DType::Float32)
            {
                throw std::runtime_error(
                    "cuDNN tensor descriptor expects Float32 tensor");
            }

            const cudnnStatus_t create_status =
                cudnnCreateTensorDescriptor(&descriptor_);

            if (create_status != CUDNN_STATUS_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to create cuDNN tensor descriptor");
            }

            const int n =
                static_cast<int>(tensor.shape()[0]);

            const int c =
                static_cast<int>(tensor.shape()[1]);

            const int h =
                static_cast<int>(tensor.shape()[2]);

            const int w =
                static_cast<int>(tensor.shape()[3]);

            const cudnnStatus_t set_status =
                cudnnSetTensor4dDescriptor(
                    descriptor_,
                    CUDNN_TENSOR_NCHW,
                    CUDNN_DATA_FLOAT,
                    n,
                    c,
                    h,
                    w);

            if (set_status != CUDNN_STATUS_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to set cuDNN tensor descriptor");
            }
        }

        ~CudnnTensorDescriptor()
        {
            if (descriptor_ != nullptr)
            {
                (void)cudnnDestroyTensorDescriptor(
                    descriptor_);
            }
        }

        CudnnTensorDescriptor(
            const CudnnTensorDescriptor &) = delete;

        CudnnTensorDescriptor &operator=(
            const CudnnTensorDescriptor &) = delete;

        CudnnTensorDescriptor(
            CudnnTensorDescriptor &&) = delete;

        CudnnTensorDescriptor &operator=(
            CudnnTensorDescriptor &&) = delete;

        [[nodiscard]] cudnnTensorDescriptor_t get() const
        {
            return descriptor_;
        }

    private:
        cudnnTensorDescriptor_t descriptor_ =
            nullptr;
    };

    class CudnnPoolingDescriptor
    {
    public:
        CudnnPoolingDescriptor(
            std::size_t kernel_height,
            std::size_t kernel_width,
            std::size_t padding_height,
            std::size_t padding_width,
            std::size_t stride_height,
            std::size_t stride_width)
        {
            const cudnnStatus_t create_status =
                cudnnCreatePoolingDescriptor(
                    &descriptor_);

            if (create_status != CUDNN_STATUS_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to create cuDNN pooling descriptor");
            }

            const cudnnStatus_t set_status =
                cudnnSetPooling2dDescriptor(
                    descriptor_,
                    CUDNN_POOLING_MAX,
                    CUDNN_PROPAGATE_NAN,
                    static_cast<int>(kernel_height),
                    static_cast<int>(kernel_width),
                    static_cast<int>(padding_height),
                    static_cast<int>(padding_width),
                    static_cast<int>(stride_height),
                    static_cast<int>(stride_width));

            if (set_status != CUDNN_STATUS_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to set cuDNN pooling descriptor");
            }
        }

        ~CudnnPoolingDescriptor()
        {
            if (descriptor_ != nullptr)
            {
                (void)cudnnDestroyPoolingDescriptor(
                    descriptor_);
            }
        }

        CudnnPoolingDescriptor(
            const CudnnPoolingDescriptor &) = delete;

        CudnnPoolingDescriptor &operator=(
            const CudnnPoolingDescriptor &) = delete;

        CudnnPoolingDescriptor(
            CudnnPoolingDescriptor &&) = delete;

        CudnnPoolingDescriptor &operator=(
            CudnnPoolingDescriptor &&) = delete;

        [[nodiscard]] cudnnPoolingDescriptor_t get() const
        {
            return descriptor_;
        }

    private:
        cudnnPoolingDescriptor_t descriptor_ =
            nullptr;
    };

    inline void cudnnMaxPool2dForwardFloat32(
        CudnnHandle &handle,
        const CudnnTensorDescriptor &input_descriptor,
        const CudnnTensorDescriptor &output_descriptor,
        const CudnnPoolingDescriptor &pooling_descriptor,
        const Tensor &input,
        Tensor &output)
    {
        const float alpha =
            1.0F;

        const float beta =
            0.0F;

        const cudnnStatus_t status =
            cudnnPoolingForward(
                handle.get(),
                pooling_descriptor.get(),
                &alpha,
                input_descriptor.get(),
                input.data(),
                &beta,
                output_descriptor.get(),
                output.data());

        if (status != CUDNN_STATUS_SUCCESS)
        {
            throw std::runtime_error(
                "cuDNN MaxPool2d forward failed");
        }
    }

    inline Tensor cudnnMaxPool2dForwardFloat32(
        const Tensor &input,
        std::size_t kernel_height,
        std::size_t kernel_width,
        std::size_t padding_height,
        std::size_t padding_width,
        std::size_t stride_height,
        std::size_t stride_width)
    {
        const std::size_t output_height =
            (input.shape()[2] + 2 * padding_height - kernel_height) /
                stride_height +
            1;

        const std::size_t output_width =
            (input.shape()[3] + 2 * padding_width - kernel_width) /
                stride_width +
            1;

        Tensor output(
            Shape{
                input.shape()[0],
                input.shape()[1],
                output_height,
                output_width},
            DType::Float32,
            Device::cuda());

        CudnnHandle handle;

        CudnnTensorDescriptor input_descriptor(
            input);

        CudnnTensorDescriptor output_descriptor(
            output);

        CudnnPoolingDescriptor pooling_descriptor(
            kernel_height,
            kernel_width,
            padding_height,
            padding_width,
            stride_height,
            stride_width);

        cudnnMaxPool2dForwardFloat32(
            handle,
            input_descriptor,
            output_descriptor,
            pooling_descriptor,
            input,
            output);

        return output;
    }

}

#endif // KL_ENABLE_CUDA

#endif // KL_TEST_CUDNN_MAXPOOL2D_HPP