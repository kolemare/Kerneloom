#include <ops/conv2d_naive.hpp>

#include <kernels/cpu/conv2d_cpu_float32_naive.hpp>

#if defined(KL_ENABLE_CUDA)
#include <kernels/cuda/conv2d_cuda_float32_naive.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <kernels/rocm/conv2d_rocm_float32_naive.hiph>
#endif

#include <stdexcept>

namespace kl
{

    // input N, C, H, W
    // kernels K, C, R, S

    namespace
    {

        void validate_conv2d_inputs(const Tensor &input, const Tensor &kernels)
        {
            if (input.device().type() != kernels.device().type())
            {
                throw std::runtime_error("conv2d expects both tensors on the same device");
            }

            if (input.dtype() != input.dtype())
            {
                throw std::runtime_error("conv2d expects tensors with the same dtype");
            }

            if (input.dtype() != DType::Float32)
            {
                throw std::runtime_error("conv2d currently supports only Float32 tensors");
            }

            if (input.rank() != 4 || kernels.rank() != 4)
            {
                throw std::runtime_error("conv2d expects two rank-2 tensors");
            }

            if (input.storage() != Storage::RowMajor ||
                input.storage() != Storage::RowMajor)
            {
                throw std::runtime_error("conv2d currently supports only RowMajor tensors");
            }
        }

    }

    Tensor conv2d_naive(const Tensor &input, const Tensor &kernels)
    {
        validate_conv2d_inputs(input, kernels);

        const std::size_t N = input.shape()[0];
        const std::size_t K = kernels.shape()[0];

        const std::size_t OH = input.shape()[2] - kernels.shape()[2] + 1;
        const std::size_t OW = input.shape()[3] - kernels.shape()[3] + 1;

        Tensor result(
            Shape{N, K, OH, OW},
            input.dtype(),
            input.device(),
            Layout::NCHW,
            Storage::RowMajor);

        switch (input.device().type())
        {
        case DeviceType::CPU:
            conv2d_cpu_float32_naive(input, kernels, result);
            return result;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            conv2d_cuda_float32_naive(input, kernels, result);
            return result;
#else
            throw std::runtime_error("CUDA conv2d requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            conv2d_rocm_float32_naive(input, kernels, result);
            return result;
#else
            throw std::runtime_error("ROCm conv2d requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error("unknown DeviceType in conv2d");
        }
    }

}