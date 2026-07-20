#ifndef KL_TEST_CUDNN_AVGPOOL2D_HPP
#define KL_TEST_CUDNN_AVGPOOL2D_HPP

#ifdef KL_ENABLE_CUDA

#include "vendor/cuda/cudnn_maxpool2d.cuh"

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>

#include <cudnn.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace kl::test
{

    class CudnnAveragePoolingDescriptor
    {
    public:
        CudnnAveragePoolingDescriptor(
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
                    std::string("Failed to create cuDNN average pooling descriptor: ") +
                    cudnnGetErrorString(create_status));
            }

            const cudnnStatus_t set_status =
                cudnnSetPooling2dDescriptor(
                    descriptor_,
                    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
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
                    std::string("Failed to set cuDNN average pooling descriptor: ") +
                    cudnnGetErrorString(set_status));
            }
        }

        ~CudnnAveragePoolingDescriptor()
        {
            if (descriptor_ != nullptr)
            {
                (void)cudnnDestroyPoolingDescriptor(
                    descriptor_);
            }
        }

        CudnnAveragePoolingDescriptor(
            const CudnnAveragePoolingDescriptor &) = delete;

        CudnnAveragePoolingDescriptor &operator=(
            const CudnnAveragePoolingDescriptor &) = delete;

        CudnnAveragePoolingDescriptor(
            CudnnAveragePoolingDescriptor &&) = delete;

        CudnnAveragePoolingDescriptor &operator=(
            CudnnAveragePoolingDescriptor &&) = delete;

        [[nodiscard]] cudnnPoolingDescriptor_t get() const
        {
            return descriptor_;
        }

    private:
        cudnnPoolingDescriptor_t descriptor_ =
            nullptr;
    };

    inline void cudnnAvgPool2dForwardFloat32(
        CudnnHandle &handle,
        const CudnnTensorDescriptor &input_descriptor,
        const CudnnTensorDescriptor &output_descriptor,
        const CudnnAveragePoolingDescriptor &pooling_descriptor,
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
                std::string("cuDNN AvgPool2d forward failed: ") +
                cudnnGetErrorString(status));
        }
    }

    inline Tensor cudnnAvgPool2dForwardFloat32(
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

        CudnnAveragePoolingDescriptor pooling_descriptor(
            kernel_height,
            kernel_width,
            padding_height,
            padding_width,
            stride_height,
            stride_width);

        cudnnAvgPool2dForwardFloat32(
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

#endif // KL_TEST_CUDNN_AVGPOOL2D_HPP