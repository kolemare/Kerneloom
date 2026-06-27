#include <ops/adam_update.hpp>

#include <ops/validation/adam_update_validation.hpp>

#include <kernels/cpu/optimizers/adam_update_cpu_float32.hpp>

#if defined(KL_ENABLE_CUDA)
#include <kernels/cuda/optimizers/adam_update_cuda_float32.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <kernels/rocm/optimizers/adam_update_rocm_float32.hiph>
#endif

#include <stdexcept>

namespace kl
{

    void adam_update_unchecked(
        Tensor &value,
        const Tensor &grad,
        Tensor &first_moment,
        Tensor &second_moment,
        float learning_rate,
        float beta1,
        float beta2,
        float epsilon,
        float beta1_power,
        float beta2_power)
    {
        switch (value.device().type())
        {
        case DeviceType::CPU:
            adam_update_cpu_float32(
                value,
                grad,
                first_moment,
                second_moment,
                learning_rate,
                beta1,
                beta2,
                epsilon,
                beta1_power,
                beta2_power);
            return;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            adam_update_cuda_float32(
                value,
                grad,
                first_moment,
                second_moment,
                learning_rate,
                beta1,
                beta2,
                epsilon,
                beta1_power,
                beta2_power);
            return;
#else
            throw std::runtime_error(
                "CUDA adam_update requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            adam_update_rocm_float32(
                value,
                grad,
                first_moment,
                second_moment,
                learning_rate,
                beta1,
                beta2,
                epsilon,
                beta1_power,
                beta2_power);
            return;
#else
            throw std::runtime_error(
                "ROCm adam_update requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error(
                "unknown DeviceType in adam_update");
        }
    }

    void adam_update(
        Tensor &value,
        const Tensor &grad,
        Tensor &first_moment,
        Tensor &second_moment,
        float learning_rate,
        float beta1,
        float beta2,
        float epsilon,
        float beta1_power,
        float beta2_power)
    {
        validate_adam_update_inputs(
            value,
            grad,
            first_moment,
            second_moment,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            beta1_power,
            beta2_power);

        adam_update_unchecked(
            value,
            grad,
            first_moment,
            second_moment,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            beta1_power,
            beta2_power);
    }

}