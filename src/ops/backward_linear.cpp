#include <ops/backward_linear.hpp>

#include <ops/validation/backward_linear_validation.hpp>

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

    void backward_linear_unchecked(
        const Tensor &input,
        const Tensor &weights,
        const Tensor &grad_output,
        Tensor &grad_input,
        Tensor &grad_weights,
        Tensor *grad_bias)
    {
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
            throw std::runtime_error(
                "CUDA backward_linear requested but CUDA backend is not enabled");
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
            throw std::runtime_error(
                "ROCm backward_linear requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error(
                "unknown DeviceType in backward_linear");
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

        backward_linear_unchecked(
            input,
            weights,
            grad_output,
            grad_input,
            grad_weights,
            grad_bias);
    }

}