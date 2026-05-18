#include <ops/backward_activation.hpp>

#include <kernels/cpu/activation/backward_relu_cpu_float32.hpp>
#include <kernels/cpu/activation/backward_sigmoid_cpu_float32.hpp>
#include <kernels/cpu/activation/backward_tanh_cpu_float32.hpp>

#if defined(KL_ENABLE_CUDA)
#include <kernels/cuda/activation/backward_relu_cuda_float32.cuh>
#include <kernels/cuda/activation/backward_sigmoid_cuda_float32.cuh>
#include <kernels/cuda/activation/backward_tanh_cuda_float32.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <kernels/rocm/activation/backward_relu_rocm_float32.hiph>
#include <kernels/rocm/activation/backward_sigmoid_rocm_float32.hiph>
#include <kernels/rocm/activation/backward_tanh_rocm_float32.hiph>
#endif

#include <stdexcept>

namespace kl
{

    namespace
    {

        void validate_backward_activation_inputs(
            const Tensor &activation_output,
            const Tensor &grad)
        {
            if (activation_output.device().type() != grad.device().type())
            {
                throw std::runtime_error("backward_activation expects tensors on the same device");
            }

            if (activation_output.dtype() != grad.dtype())
            {
                throw std::runtime_error("backward_activation expects tensors with the same dtype");
            }

            if (activation_output.dtype() != DType::Float32)
            {
                throw std::runtime_error("backward_activation currently supports only Float32 tensors");
            }

            if (activation_output.storage() != Storage::RowMajor ||
                grad.storage() != Storage::RowMajor)
            {
                throw std::runtime_error("backward_activation currently supports only RowMajor tensors");
            }

            if (activation_output.shape() != grad.shape())
            {
                throw std::runtime_error("backward_activation expects tensors with the same shape");
            }
        }

    }

    void backward_activation(
        const Tensor &activation_output,
        Tensor &grad,
        ActivationType type)
    {
        validate_backward_activation_inputs(
            activation_output,
            grad);

        switch (activation_output.device().type())
        {
        case DeviceType::CPU:
            switch (type)
            {
            case ActivationType::ReLU:
                backward_relu_cpu_float32(activation_output, grad);
                return;

            case ActivationType::Sigmoid:
                backward_sigmoid_cpu_float32(activation_output, grad);
                return;

            case ActivationType::Tanh:
                backward_tanh_cpu_float32(activation_output, grad);
                return;

            default:
                throw std::runtime_error("unknown ActivationType in CPU backward_activation");
            }

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            switch (type)
            {
            case ActivationType::ReLU:
                backward_relu_cuda_float32(activation_output, grad);
                return;

            case ActivationType::Sigmoid:
                backward_sigmoid_cuda_float32(activation_output, grad);
                return;

            case ActivationType::Tanh:
                backward_tanh_cuda_float32(activation_output, grad);
                return;

            default:
                throw std::runtime_error("unknown ActivationType in CUDA backward_activation");
            }
#else
            throw std::runtime_error("CUDA backward_activation requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            switch (type)
            {
            case ActivationType::ReLU:
                backward_relu_rocm_float32(activation_output, grad);
                return;

            case ActivationType::Sigmoid:
                backward_sigmoid_rocm_float32(activation_output, grad);
                return;

            case ActivationType::Tanh:
                backward_tanh_rocm_float32(activation_output, grad);
                return;

            default:
                throw std::runtime_error("unknown ActivationType in ROCm backward_activation");
            }
#else
            throw std::runtime_error("ROCm backward_activation requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error("unknown DeviceType in backward_activation");
        }
    }

}