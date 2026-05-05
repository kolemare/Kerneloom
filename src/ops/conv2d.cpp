#include <ops/conv2d.hpp>

#include <kernels/cpu/conv2d_cpu_float32.hpp>

#if defined(KL_ENABLE_CUDA)
#include <kernels/cuda/conv2d_cuda_float32.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <kernels/rocm/conv2d_rocm_float32.hiph>
#endif

#include <stdexcept>

namespace kl
{

    namespace
    {

        void validate_conv2d_inputs(
            const Tensor &input,
            const Tensor &kernels,
            const Tensor *bias,
            const Tensor &result,
            const Conv2dOptions &options)
        {
            if (options.stride_h == 0 || options.stride_w == 0)
            {
                throw std::runtime_error("conv2d stride must be greater than zero");
            }

            if (options.dilation_h == 0 || options.dilation_w == 0)
            {
                throw std::runtime_error("conv2d dilation must be greater than zero");
            }

            if (input.device().type() != kernels.device().type() ||
                input.device().type() != result.device().type())
            {
                throw std::runtime_error("conv2d expects input, kernels, and result on the same device");
            }

            if (bias != nullptr && bias->device().type() != input.device().type())
            {
                throw std::runtime_error("conv2d expects bias on the same device as input");
            }

            if (input.dtype() != kernels.dtype() ||
                input.dtype() != result.dtype())
            {
                throw std::runtime_error("conv2d expects input, kernels, and result with the same dtype");
            }

            if (bias != nullptr && bias->dtype() != input.dtype())
            {
                throw std::runtime_error("conv2d expects bias with the same dtype as input");
            }

            if (input.dtype() != DType::Float32)
            {
                throw std::runtime_error("conv2d currently supports only Float32 tensors");
            }

            if (input.rank() != 4)
            {
                throw std::runtime_error("conv2d expects input shape N x C x H x W");
            }

            if (kernels.rank() != 4)
            {
                throw std::runtime_error("conv2d expects kernels shape K x C x R x S");
            }

            if (result.rank() != 4)
            {
                throw std::runtime_error("conv2d expects result shape N x K x OH x OW");
            }

            if (bias != nullptr && bias->rank() != 1)
            {
                throw std::runtime_error("conv2d expects bias shape K");
            }

            if (input.storage() != Storage::RowMajor ||
                kernels.storage() != Storage::RowMajor ||
                result.storage() != Storage::RowMajor)
            {
                throw std::runtime_error("conv2d currently supports only RowMajor tensors");
            }

            if (bias != nullptr && bias->storage() != Storage::RowMajor)
            {
                throw std::runtime_error("conv2d currently supports only RowMajor bias");
            }

            const std::size_t N = input.shape()[0];
            const std::size_t C = input.shape()[1];
            const std::size_t H = input.shape()[2];
            const std::size_t W = input.shape()[3];

            const std::size_t K = kernels.shape()[0];
            const std::size_t KC = kernels.shape()[1];
            const std::size_t R = kernels.shape()[2];
            const std::size_t S = kernels.shape()[3];

            if (C != KC)
            {
                throw std::runtime_error("conv2d input channels must match kernel channels");
            }

            if (bias != nullptr && bias->shape()[0] != K)
            {
                throw std::runtime_error("conv2d bias size must match output channels");
            }

            const std::size_t effective_r =
                options.dilation_h * (R - 1) + 1;

            const std::size_t effective_s =
                options.dilation_w * (S - 1) + 1;

            if (effective_r > H + 2 * options.padding_h ||
                effective_s > W + 2 * options.padding_w)
            {
                throw std::runtime_error("conv2d effective kernel cannot be larger than padded input");
            }

            const std::size_t OH =
                (H + 2 * options.padding_h - effective_r) / options.stride_h + 1;

            const std::size_t OW =
                (W + 2 * options.padding_w - effective_s) / options.stride_w + 1;

            if (result.shape()[0] != N ||
                result.shape()[1] != K ||
                result.shape()[2] != OH ||
                result.shape()[3] != OW)
            {
                throw std::runtime_error("conv2d result tensor has incorrect shape");
            }
        }

    }

    void conv2d(
        const Tensor &input,
        const Tensor &kernels,
        const Tensor *bias,
        Tensor &result,
        const Conv2dOptions &options)
    {
        validate_conv2d_inputs(input, kernels, bias, result, options);

        switch (input.device().type())
        {
        case DeviceType::CPU:
            conv2d_cpu_float32(input, kernels, bias, result, options);
            return;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            conv2d_cuda_float32(input, kernels, bias, result, options);
            return;
#else
            throw std::runtime_error("CUDA conv2d requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            conv2d_rocm_float32(input, kernels, bias, result, options);
            return;
#else
            throw std::runtime_error("ROCm conv2d requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error("unknown DeviceType in conv2d");
        }
    }

}