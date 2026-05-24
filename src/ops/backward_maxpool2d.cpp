#include <ops/backward_maxpool2d.hpp>

#include <kernels/cpu/pooling/backward_maxpool2d_cpu_float32.hpp>

#if defined(KL_ENABLE_CUDA)
#include <kernels/cuda/pooling/backward_maxpool2d_cuda_float32.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <kernels/rocm/pooling/backward_maxpool2d_rocm_float32.hiph>
#endif

#include <stdexcept>

namespace kl
{

    namespace
    {

        void validate_backward_maxpool2d_inputs(
            const Tensor &indices,
            const Tensor &grad_output,
            const Tensor &grad_input)
        {
            if (indices.device().type() != grad_output.device().type() ||
                indices.device().type() != grad_input.device().type())
            {
                throw std::runtime_error("backward_maxpool2d expects tensors on the same device");
            }

            if (grad_output.dtype() != grad_input.dtype())
            {
                throw std::runtime_error("backward_maxpool2d expects grad_output and grad_input with the same dtype");
            }

            if (grad_output.dtype() != DType::Float32)
            {
                throw std::runtime_error("backward_maxpool2d currently supports only Float32 gradients");
            }

            if (indices.dtype() != DType::Int32)
            {
                throw std::runtime_error("backward_maxpool2d expects Int32 indices tensor");
            }

            if (indices.rank() != 4)
            {
                throw std::runtime_error("backward_maxpool2d expects indices shape N x C x OH x OW");
            }

            if (grad_output.rank() != 4)
            {
                throw std::runtime_error("backward_maxpool2d expects grad_output shape N x C x OH x OW");
            }

            if (grad_input.rank() != 4)
            {
                throw std::runtime_error("backward_maxpool2d expects grad_input shape N x C x H x W");
            }

            if (indices.shape() != grad_output.shape())
            {
                throw std::runtime_error("backward_maxpool2d indices shape must match grad_output shape");
            }

            if (indices.shape()[0] != grad_input.shape()[0] ||
                indices.shape()[1] != grad_input.shape()[1])
            {
                throw std::runtime_error("backward_maxpool2d N/C shape mismatch");
            }

            if (indices.storage() != Storage::RowMajor ||
                grad_output.storage() != Storage::RowMajor ||
                grad_input.storage() != Storage::RowMajor)
            {
                throw std::runtime_error("backward_maxpool2d currently supports only RowMajor tensors");
            }
        }

    }

    void backward_maxpool2d(
        const Tensor &indices,
        const Tensor &grad_output,
        Tensor &grad_input)
    {
        validate_backward_maxpool2d_inputs(
            indices,
            grad_output,
            grad_input);

        switch (grad_output.device().type())
        {
        case DeviceType::CPU:
            backward_maxpool2d_cpu_float32(
                indices,
                grad_output,
                grad_input);
            return;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            backward_maxpool2d_cuda_float32(
                indices,
                grad_output,
                grad_input);
            return;
#else
            throw std::runtime_error("CUDA backward_maxpool2d requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            backward_maxpool2d_rocm_float32(
                indices,
                grad_output,
                grad_input);
            return;
#else
            throw std::runtime_error("ROCm backward_maxpool2d requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error("unknown DeviceType in backward_maxpool2d");
        }
    }

}