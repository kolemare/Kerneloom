#include <ops/backward_avgpool2d.hpp>

#include <ops/validation/backward_avgpool2d_validation.hpp>

#include <kernels/cpu/pooling/backward_avgpool2d_cpu_float32.hpp>

#if defined(KL_ENABLE_CUDA)
#include <kernels/cuda/pooling/backward_avgpool2d_cuda_float32.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <kernels/rocm/pooling/backward_avgpool2d_rocm_float32.hiph>
#endif

#include <stdexcept>

namespace kl
{

    void backward_avgpool2d(
        const Tensor &grad_output,
        Tensor &grad_input,
        const Pooling2dOptions &options)
    {
        validate_backward_avgpool2d_inputs(
            grad_output,
            grad_input,
            options);

        switch (grad_output.device().type())
        {
        case DeviceType::CPU:
            backward_avgpool2d_cpu_float32(
                grad_output,
                grad_input,
                options);
            return;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            backward_avgpool2d_cuda_float32(
                grad_output,
                grad_input,
                options);
            return;
#else
            throw std::runtime_error(
                "CUDA backward_avgpool2d requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            backward_avgpool2d_rocm_float32(
                grad_output,
                grad_input,
                options);
            return;
#else
            throw std::runtime_error(
                "ROCm backward_avgpool2d requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error(
                "unknown DeviceType in backward_avgpool2d");
        }
    }

}