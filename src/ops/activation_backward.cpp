#include <ops/activation_backward.hpp>

#include <kernels/cpu/activation/relu_backward_cpu_float32.hpp>
#include <kernels/cpu/activation/sigmoid_backward_cpu_float32.hpp>
#include <kernels/cpu/activation/tanh_backward_cpu_float32.hpp>

#if defined(KL_ENABLE_CUDA)
#include <kernels/cuda/activation/relu_backward_cuda_float32.cuh>
#include <kernels/cuda/activation/sigmoid_backward_cuda_float32.cuh>
#include <kernels/cuda/activation/tanh_backward_cuda_float32.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <kernels/rocm/activation/relu_backward_rocm_float32.hiph>
#include <kernels/rocm/activation/sigmoid_backward_rocm_float32.hiph>
#include <kernels/rocm/activation/tanh_backward_rocm_float32.hiph>
#endif

#include <stdexcept>

namespace kl
{

    namespace
    {

        void validate_activation_backward_inputs(
            const Tensor &activation_output,
            const Tensor &grad)
        {
            if (activation_output.device().type() != grad.device().type())
            {
                throw std::runtime_error("activation_backward expects tensors on the same device");
            }

            if (activation_output.dtype() != grad.dtype())
            {
                throw std::runtime_error("activation_backward expects tensors with the same dtype");
            }

            if (activation_output.dtype() != DType::Float32)
            {
                throw std::runtime_error("activation_backward currently supports only Float32 tensors");
            }

            if (activation_output.storage() != Storage::RowMajor ||
                grad.storage() != Storage::RowMajor)
            {
                throw std::runtime_error("activation_backward currently supports only RowMajor tensors");
            }

            if (activation_output.shape() != grad.shape())
            {
                throw std::runtime_error("activation_backward expects tensors with the same shape");
            }
        }

    }

    void activation_backward(
        const Tensor &activation_output,
        Tensor &grad,
        ActivationType type)
    {
        validate_activation_backward_inputs(
            activation_output,
            grad);

        switch (activation_output.device().type())
        {
        case DeviceType::CPU:
            switch (type)
            {
            case ActivationType::ReLU:
                relu_backward_cpu_float32(activation_output, grad);
                return;

            case ActivationType::Sigmoid:
                sigmoid_backward_cpu_float32(activation_output, grad);
                return;

            case ActivationType::Tanh:
                tanh_backward_cpu_float32(activation_output, grad);
                return;

            default:
                throw std::runtime_error("unknown ActivationType in CPU activation_backward");
            }

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            switch (type)
            {
            case ActivationType::ReLU:
                relu_backward_cuda_float32(activation_output, grad);
                return;

            case ActivationType::Sigmoid:
                sigmoid_backward_cuda_float32(activation_output, grad);
                return;

            case ActivationType::Tanh:
                tanh_backward_cuda_float32(activation_output, grad);
                return;

            default:
                throw std::runtime_error("unknown ActivationType in CUDA activation_backward");
            }
#else
            throw std::runtime_error("CUDA activation_backward requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            switch (type)
            {
            case ActivationType::ReLU:
                relu_backward_rocm_float32(activation_output, grad);
                return;

            case ActivationType::Sigmoid:
                sigmoid_backward_rocm_float32(activation_output, grad);
                return;

            case ActivationType::Tanh:
                tanh_backward_rocm_float32(activation_output, grad);
                return;

            default:
                throw std::runtime_error("unknown ActivationType in ROCm activation_backward");
            }
#else
            throw std::runtime_error("ROCm activation_backward requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error("unknown DeviceType in activation_backward");
        }
    }

}