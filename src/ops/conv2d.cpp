#include <ops/conv2d.hpp>

#include <ops/validation/conv2d_validation.hpp>

#include <kernels/cpu/conv2d/conv2d_cpu_float32_general.hpp>
#include <kernels/cpu/conv2d/conv2d_cpu_float32_padded.hpp>
#include <kernels/cpu/conv2d/conv2d_cpu_float32_strided.hpp>
#include <kernels/cpu/conv2d/conv2d_cpu_float32_valid.hpp>

#if defined(KL_ENABLE_CUDA)
#include <kernels/cuda/conv2d/conv2d_cuda_float32_general.cuh>
#include <kernels/cuda/conv2d/conv2d_cuda_float32_padded.cuh>
#include <kernels/cuda/conv2d/conv2d_cuda_float32_strided.cuh>
#include <kernels/cuda/conv2d/conv2d_cuda_float32_valid.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <kernels/rocm/conv2d/conv2d_rocm_float32_general.hiph>
#include <kernels/rocm/conv2d/conv2d_rocm_float32_padded.hiph>
#include <kernels/rocm/conv2d/conv2d_rocm_float32_strided.hiph>
#include <kernels/rocm/conv2d/conv2d_rocm_float32_valid.hiph>
#endif

#include <stdexcept>

namespace kl
{

    namespace
    {

        bool is_valid_conv2d(
            const Conv2dOptions &options)
        {
            return options.stride_h == 1 &&
                   options.stride_w == 1 &&
                   options.padding_h == 0 &&
                   options.padding_w == 0 &&
                   options.dilation_h == 1 &&
                   options.dilation_w == 1;
        }

        bool is_padded_conv2d(
            const Conv2dOptions &options)
        {
            return options.stride_h == 1 &&
                   options.stride_w == 1 &&
                   (options.padding_h > 0 ||
                    options.padding_w > 0) &&
                   options.dilation_h == 1 &&
                   options.dilation_w == 1;
        }

        bool is_strided_conv2d(
            const Conv2dOptions &options)
        {
            return (options.stride_h > 1 ||
                    options.stride_w > 1) &&
                   options.padding_h == 0 &&
                   options.padding_w == 0 &&
                   options.dilation_h == 1 &&
                   options.dilation_w == 1;
        }

        void conv2d_cpu_float32(
            const Tensor &input,
            const Tensor &kernels,
            const Tensor *bias,
            Tensor &result,
            const Conv2dOptions &options)
        {
            if (is_valid_conv2d(
                    options))
            {
                conv2d_cpu_float32_valid(
                    input,
                    kernels,
                    bias,
                    result,
                    options);
                return;
            }

            if (is_padded_conv2d(
                    options))
            {
                conv2d_cpu_float32_padded(
                    input,
                    kernels,
                    bias,
                    result,
                    options);
                return;
            }

            if (is_strided_conv2d(
                    options))
            {
                conv2d_cpu_float32_strided(
                    input,
                    kernels,
                    bias,
                    result,
                    options);
                return;
            }

            conv2d_cpu_float32_general(
                input,
                kernels,
                bias,
                result,
                options);
        }

#if defined(KL_ENABLE_CUDA)
        void conv2d_cuda_float32(
            const Tensor &input,
            const Tensor &kernels,
            const Tensor *bias,
            Tensor &result,
            const Conv2dOptions &options)
        {
            if (is_valid_conv2d(
                    options))
            {
                conv2d_cuda_float32_valid(
                    input,
                    kernels,
                    bias,
                    result,
                    options);
                return;
            }

            if (is_padded_conv2d(
                    options))
            {
                conv2d_cuda_float32_padded(
                    input,
                    kernels,
                    bias,
                    result,
                    options);
                return;
            }

            if (is_strided_conv2d(
                    options))
            {
                conv2d_cuda_float32_strided(
                    input,
                    kernels,
                    bias,
                    result,
                    options);
                return;
            }

            conv2d_cuda_float32_general(
                input,
                kernels,
                bias,
                result,
                options);
        }
#endif

#if defined(KL_ENABLE_ROCM)
        void conv2d_rocm_float32(
            const Tensor &input,
            const Tensor &kernels,
            const Tensor *bias,
            Tensor &result,
            const Conv2dOptions &options)
        {
            if (is_valid_conv2d(
                    options))
            {
                conv2d_rocm_float32_valid(
                    input,
                    kernels,
                    bias,
                    result,
                    options);
                return;
            }

            if (is_padded_conv2d(
                    options))
            {
                conv2d_rocm_float32_padded(
                    input,
                    kernels,
                    bias,
                    result,
                    options);
                return;
            }

            if (is_strided_conv2d(
                    options))
            {
                conv2d_rocm_float32_strided(
                    input,
                    kernels,
                    bias,
                    result,
                    options);
                return;
            }

            conv2d_rocm_float32_general(
                input,
                kernels,
                bias,
                result,
                options);
        }
#endif

    }

    void conv2d_unchecked(
        const Tensor &input,
        const Tensor &kernels,
        const Tensor *bias,
        Tensor &result,
        const Conv2dOptions &options)
    {
        switch (input.device().type())
        {
        case DeviceType::CPU:
            conv2d_cpu_float32(
                input,
                kernels,
                bias,
                result,
                options);
            return;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            conv2d_cuda_float32(
                input,
                kernels,
                bias,
                result,
                options);
            return;
#else
            throw std::runtime_error(
                "CUDA conv2d requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            conv2d_rocm_float32(
                input,
                kernels,
                bias,
                result,
                options);
            return;
#else
            throw std::runtime_error(
                "ROCm conv2d requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error(
                "unknown DeviceType in conv2d");
        }
    }

    void conv2d(
        const Tensor &input,
        const Tensor &kernels,
        const Tensor *bias,
        Tensor &result,
        const Conv2dOptions &options)
    {
        validate_conv2d_inputs(
            input,
            kernels,
            bias,
            result,
            options);

        conv2d_unchecked(
            input,
            kernels,
            bias,
            result,
            options);
    }

}