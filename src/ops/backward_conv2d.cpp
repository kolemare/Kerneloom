#include <ops/backward_conv2d.hpp>

#include <ops/validation/backward_conv2d_validation.hpp>

#include <kernels/cpu/conv2d/backward_conv2d_cpu_float32.hpp>

#if defined(KL_ENABLE_CUDA)
#include <kernels/cuda/conv2d/backward_conv2d_cuda_float32.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <kernels/rocm/conv2d/backward_conv2d_rocm_float32.hiph>
#endif

#include <stdexcept>

namespace kl
{

    void backward_conv2d(
        const Tensor &input,
        const Tensor &weights,
        const Tensor &grad_output,
        Tensor &grad_input,
        Tensor &grad_weights,
        Tensor *grad_bias,
        const Conv2dOptions &options)
    {
        validate_backward_conv2d_inputs(
            input,
            weights,
            grad_output,
            grad_input,
            grad_weights,
            grad_bias,
            options);

        switch (input.device().type())
        {
        case DeviceType::CPU:
            backward_conv2d_cpu_float32(
                input,
                weights,
                grad_output,
                grad_input,
                grad_weights,
                grad_bias,
                options);
            return;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            backward_conv2d_cuda_float32(
                input,
                weights,
                grad_output,
                grad_input,
                grad_weights,
                grad_bias,
                options);
            return;
#else
            throw std::runtime_error(
                "CUDA backward_conv2d requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            backward_conv2d_rocm_float32(
                input,
                weights,
                grad_output,
                grad_input,
                grad_weights,
                grad_bias,
                options);
            return;
#else
            throw std::runtime_error(
                "ROCm backward_conv2d requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error(
                "unknown DeviceType in backward_conv2d");
        }
    }

}