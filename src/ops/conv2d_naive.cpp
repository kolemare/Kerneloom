#include <ops/conv2d_naive.hpp>

#include <kernels/cpu/naive/conv2d_cpu_float32_naive.hpp>

#if defined(KL_ENABLE_CUDA)
#include <kernels/cuda/naive/conv2d_cuda_float32_naive.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <kernels/rocm/naive/conv2d_rocm_float32_naive.hiph>
#endif

#include <stdexcept>

namespace kl
{

    namespace
    {

        void validate_conv2d_inputs(
            const Tensor &input,
            const Tensor &kernels,
            const Tensor &result)
        {
            if (input.device().type() != kernels.device().type() ||
                input.device().type() != result.device().type())
            {
                throw std::runtime_error("conv2d expects input, kernels, and result on the same device");
            }

            if (input.dtype() != kernels.dtype() ||
                input.dtype() != result.dtype())
            {
                throw std::runtime_error("conv2d expects input, kernels, and result with the same dtype");
            }

            if (input.dtype() != DType::Float32)
            {
                throw std::runtime_error("conv2d currently supports only Float32 tensors");
            }

            if (input.layout() != kl::Layout::NCHW || kernels.layout() != kl::Layout::NCHW)
            {
                throw std::runtime_error("conv2d expects NCHW layout for both input and kernels.");
            }

            if (input.rank() != 4)
            {
                throw std::runtime_error("conv2d expects input shape N x C x H x W");
            }

            if (kernels.rank() != 4)
            {
                throw std::runtime_error("conv2d expects kernel shape K x C x R x S");
            }

            if (result.rank() != 4)
            {
                throw std::runtime_error("conv2d expects result shape N x K x OH x OW");
            }

            if (input.storage() != Storage::RowMajor ||
                kernels.storage() != Storage::RowMajor ||
                result.storage() != Storage::RowMajor)
            {
                throw std::runtime_error("conv2d currently supports only RowMajor tensors");
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

            if (R > H || S > W)
            {
                throw std::runtime_error("conv2d kernel cannot be larger than input");
            }

            const std::size_t OH = H - R + 1;
            const std::size_t OW = W - S + 1;

            if (result.shape()[0] != N ||
                result.shape()[1] != K ||
                result.shape()[2] != OH ||
                result.shape()[3] != OW)
            {
                throw std::runtime_error("conv2d result tensor has incorrect shape");
            }
        }

    }

    void conv2d_naive(
        const Tensor &input,
        const Tensor &kernels,
        Tensor &result)
    {
        validate_conv2d_inputs(input, kernels, result);

        switch (input.device().type())
        {
        case DeviceType::CPU:
            conv2d_cpu_float32_naive(input, kernels, result);
            return;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            conv2d_cuda_float32_naive(input, kernels, result);
            return;
#else
            throw std::runtime_error("CUDA conv2d requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            conv2d_rocm_float32_naive(input, kernels, result);
            return;
#else
            throw std::runtime_error("ROCm conv2d requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error("unknown DeviceType in conv2d");
        }
    }

}