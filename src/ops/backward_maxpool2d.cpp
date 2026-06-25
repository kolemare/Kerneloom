#include <ops/backward_maxpool2d.hpp>

#include <ops/validation/backward_maxpool2d_validation.hpp>

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
            throw std::runtime_error(
                "CUDA backward_maxpool2d requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            backward_maxpool2d_rocm_float32(
                indices,
                grad_output,
                grad_input);
            return;
#else
            throw std::runtime_error(
                "ROCm backward_maxpool2d requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error(
                "unknown DeviceType in backward_maxpool2d");
        }
    }

}