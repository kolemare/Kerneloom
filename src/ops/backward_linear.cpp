#include <ops/backward_linear.hpp>

#include <kernels/cpu/linear/backward_linear_cpu_float32.hpp>

#if defined(KL_ENABLE_CUDA)
#include <kernels/cuda/linear/backward_linear_cuda_float32.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <kernels/rocm/linear/backward_linear_rocm_float32.hiph>
#endif

#include <stdexcept>

namespace kl
{

    namespace
    {

        void validate_backward_linear_inputs(
            const Tensor &input,
            const Tensor &weights,
            const Tensor &grad_output,
            const Tensor &grad_input,
            const Tensor &grad_weights,
            const Tensor *grad_bias)
        {
            if (input.device().type() != weights.device().type() ||
                input.device().type() != grad_output.device().type() ||
                input.device().type() != grad_input.device().type() ||
                input.device().type() != grad_weights.device().type())
            {
                throw std::runtime_error("backward_linear expects tensors on the same device");
            }

            if (grad_bias != nullptr &&
                grad_bias->device().type() != input.device().type())
            {
                throw std::runtime_error("backward_linear expects grad_bias on the same device");
            }

            if (input.dtype() != weights.dtype() ||
                input.dtype() != grad_output.dtype() ||
                input.dtype() != grad_input.dtype() ||
                input.dtype() != grad_weights.dtype())
            {
                throw std::runtime_error("backward_linear expects tensors with the same dtype");
            }

            if (grad_bias != nullptr &&
                grad_bias->dtype() != input.dtype())
            {
                throw std::runtime_error("backward_linear expects grad_bias with the same dtype");
            }

            if (input.dtype() != DType::Float32)
            {
                throw std::runtime_error("backward_linear currently supports only Float32 tensors");
            }

            if (input.storage() != Storage::RowMajor ||
                weights.storage() != Storage::RowMajor ||
                grad_output.storage() != Storage::RowMajor ||
                grad_input.storage() != Storage::RowMajor ||
                grad_weights.storage() != Storage::RowMajor)
            {
                throw std::runtime_error("backward_linear currently supports only RowMajor tensors");
            }

            if (grad_bias != nullptr &&
                grad_bias->storage() != Storage::RowMajor)
            {
                throw std::runtime_error("backward_linear currently supports only RowMajor grad_bias");
            }

            if (input.rank() != 2)
            {
                throw std::runtime_error("backward_linear expects input shape N x IN");
            }

            if (weights.rank() != 2)
            {
                throw std::runtime_error("backward_linear expects weights shape OUT x IN");
            }

            if (grad_output.rank() != 2)
            {
                throw std::runtime_error("backward_linear expects grad_output shape N x OUT");
            }

            if (grad_input.rank() != 2)
            {
                throw std::runtime_error("backward_linear expects grad_input shape N x IN");
            }

            if (grad_weights.rank() != 2)
            {
                throw std::runtime_error("backward_linear expects grad_weights shape OUT x IN");
            }

            const std::size_t batch_size = input.shape()[0];
            const std::size_t input_features = input.shape()[1];
            const std::size_t output_features = weights.shape()[0];

            if (weights.shape()[1] != input_features)
            {
                throw std::runtime_error("backward_linear input/weight feature mismatch");
            }

            if (grad_output.shape()[0] != batch_size ||
                grad_output.shape()[1] != output_features)
            {
                throw std::runtime_error("backward_linear grad_output shape mismatch");
            }

            if (grad_input.shape()[0] != batch_size ||
                grad_input.shape()[1] != input_features)
            {
                throw std::runtime_error("backward_linear grad_input shape mismatch");
            }

            if (grad_weights.shape()[0] != output_features ||
                grad_weights.shape()[1] != input_features)
            {
                throw std::runtime_error("backward_linear grad_weights shape mismatch");
            }

            if (grad_bias != nullptr)
            {
                if (grad_bias->rank() != 1 ||
                    grad_bias->shape()[0] != output_features)
                {
                    throw std::runtime_error("backward_linear grad_bias shape mismatch");
                }
            }
        }

    }

    void backward_linear(
        const Tensor &input,
        const Tensor &weights,
        const Tensor &grad_output,
        Tensor &grad_input,
        Tensor &grad_weights,
        Tensor *grad_bias)
    {
        validate_backward_linear_inputs(
            input,
            weights,
            grad_output,
            grad_input,
            grad_weights,
            grad_bias);

        switch (input.device().type())
        {
        case DeviceType::CPU:
            backward_linear_cpu_float32(
                input,
                weights,
                grad_output,
                grad_input,
                grad_weights,
                grad_bias);
            return;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            backward_linear_cuda_float32(
                input,
                weights,
                grad_output,
                grad_input,
                grad_weights,
                grad_bias);
            return;
#else
            throw std::runtime_error("CUDA backward_linear requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            backward_linear_rocm_float32(
                input,
                weights,
                grad_output,
                grad_input,
                grad_weights,
                grad_bias);
            return;
#else
            throw std::runtime_error("ROCm backward_linear requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error("unknown DeviceType in backward_linear");
        }
    }

}