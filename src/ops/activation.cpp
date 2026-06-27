#include <ops/activation.hpp>

#include <ops/validation/activation_validation.hpp>

#include <kernels/cpu/activation/relu_cpu_float32.hpp>
#include <kernels/cpu/activation/sigmoid_cpu_float32.hpp>
#include <kernels/cpu/activation/softmax_cpu_float32.hpp>
#include <kernels/cpu/activation/tanh_cpu_float32.hpp>

#if defined(KL_ENABLE_CUDA)
#include <kernels/cuda/activation/relu_cuda_float32.cuh>
#include <kernels/cuda/activation/sigmoid_cuda_float32.cuh>
#include <kernels/cuda/activation/softmax_cuda_float32.cuh>
#include <kernels/cuda/activation/tanh_cuda_float32.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <kernels/rocm/activation/relu_rocm_float32.hiph>
#include <kernels/rocm/activation/sigmoid_rocm_float32.hiph>
#include <kernels/rocm/activation/softmax_rocm_float32.hiph>
#include <kernels/rocm/activation/tanh_rocm_float32.hiph>
#endif

#include <stdexcept>

namespace kl
{

    const char *activation_type_name(
        ActivationType type)
    {
        switch (type)
        {
        case ActivationType::ReLU:
            return "ReLU";

        case ActivationType::Sigmoid:
            return "Sigmoid";

        case ActivationType::Tanh:
            return "Tanh";

        case ActivationType::Softmax:
            return "Softmax";

        default:
            return "Unknown";
        }
    }

    namespace
    {

        void activation_cpu_float32(
            Tensor &tensor,
            ActivationType type)
        {
            switch (type)
            {
            case ActivationType::ReLU:
                relu_cpu_float32(
                    tensor);
                return;

            case ActivationType::Sigmoid:
                sigmoid_cpu_float32(
                    tensor);
                return;

            case ActivationType::Tanh:
                tanh_cpu_float32(
                    tensor);
                return;

            case ActivationType::Softmax:
                softmax_cpu_float32(
                    tensor);
                return;

            default:
                throw std::runtime_error(
                    "unknown CPU activation type");
            }
        }

#if defined(KL_ENABLE_CUDA)
        void activation_cuda_float32(
            Tensor &tensor,
            ActivationType type)
        {
            switch (type)
            {
            case ActivationType::ReLU:
                relu_cuda_float32(
                    tensor);
                return;

            case ActivationType::Sigmoid:
                sigmoid_cuda_float32(
                    tensor);
                return;

            case ActivationType::Tanh:
                tanh_cuda_float32(
                    tensor);
                return;

            case ActivationType::Softmax:
                softmax_cuda_float32(
                    tensor);
                return;

            default:
                throw std::runtime_error(
                    "unknown CUDA activation type");
            }
        }
#endif

#if defined(KL_ENABLE_ROCM)
        void activation_rocm_float32(
            Tensor &tensor,
            ActivationType type)
        {
            switch (type)
            {
            case ActivationType::ReLU:
                relu_rocm_float32(
                    tensor);
                return;

            case ActivationType::Sigmoid:
                sigmoid_rocm_float32(
                    tensor);
                return;

            case ActivationType::Tanh:
                tanh_rocm_float32(
                    tensor);
                return;

            case ActivationType::Softmax:
                softmax_rocm_float32(
                    tensor);
                return;

            default:
                throw std::runtime_error(
                    "unknown ROCm activation type");
            }
        }
#endif

    }

    void activation_unchecked(
        Tensor &tensor,
        ActivationType type)
    {
        switch (tensor.device().type())
        {
        case DeviceType::CPU:
            activation_cpu_float32(
                tensor,
                type);
            return;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            activation_cuda_float32(
                tensor,
                type);
            return;
#else
            throw std::runtime_error(
                "CUDA activation requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            activation_rocm_float32(
                tensor,
                type);
            return;
#else
            throw std::runtime_error(
                "ROCm activation requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error(
                "unknown DeviceType in activation");
        }
    }

    void activation(
        Tensor &tensor,
        ActivationType type)
    {
        validate_activation_tensor(
            tensor,
            type);

        activation_unchecked(
            tensor,
            type);
    }

}