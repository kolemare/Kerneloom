#include <ops/sgd_update.hpp>

#include <ops/validation/sgd_update_validation.hpp>

#include <kernels/cpu/optimizers/sgd_update_cpu_float32.hpp>

#if defined(KL_ENABLE_CUDA)
#include <kernels/cuda/optimizers/sgd_update_cuda_float32.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <kernels/rocm/optimizers/sgd_update_rocm_float32.hiph>
#endif

#include <stdexcept>

namespace kl
{

    void sgd_update(
        Tensor &value,
        const Tensor &grad,
        float learning_rate)
    {
        validate_sgd_update_inputs(
            value,
            grad);

        switch (value.device().type())
        {
        case DeviceType::CPU:
            sgd_update_cpu_float32(
                value,
                grad,
                learning_rate);
            return;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            sgd_update_cuda_float32(
                value,
                grad,
                learning_rate);
            return;
#else
            throw std::runtime_error(
                "CUDA sgd_update requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            sgd_update_rocm_float32(
                value,
                grad,
                learning_rate);
            return;
#else
            throw std::runtime_error(
                "ROCm sgd_update requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error(
                "unknown DeviceType in sgd_update");
        }
    }

}