#include <ops/categorical_cross_entropy.hpp>

#include <kernels/cpu/losses/categorical_cross_entropy_cpu_float32.hpp>

#if defined(KL_ENABLE_CUDA)
#include <kernels/cuda/losses/categorical_cross_entropy_cuda_float32.cuh>
#endif

#if defined(KL_ENABLE_ROCM)
#include <kernels/rocm/losses/categorical_cross_entropy_rocm_float32.hiph>
#endif

#include <stdexcept>

namespace kl
{

    namespace
    {

        void validate_categorical_cross_entropy_inputs(
            const Tensor &prediction,
            const Tensor &target,
            const Tensor &result,
            Reduction reduction,
            std::size_t valid_sample_count)
        {
            if (prediction.device().type() !=
                    target.device().type() ||
                prediction.device().type() !=
                    result.device().type())
            {
                throw std::runtime_error(
                    "categorical_cross_entropy expects tensors on the same device");
            }

            if (prediction.dtype() !=
                DType::Float32)
            {
                throw std::runtime_error(
                    "categorical_cross_entropy expects Float32 prediction");
            }

            if (target.dtype() !=
                DType::Int32)
            {
                throw std::runtime_error(
                    "categorical_cross_entropy expects Int32 target");
            }

            if (result.dtype() !=
                DType::Float32)
            {
                throw std::runtime_error(
                    "categorical_cross_entropy expects Float32 result");
            }

            if (prediction.rank() !=
                2)
            {
                throw std::runtime_error(
                    "categorical_cross_entropy expects prediction shape N x C");
            }

            if (prediction.shape()[0] ==
                    0 ||
                prediction.shape()[1] ==
                    0)
            {
                throw std::runtime_error(
                    "categorical_cross_entropy expects non-empty prediction");
            }

            if (target.rank() !=
                1)
            {
                throw std::runtime_error(
                    "categorical_cross_entropy expects target shape N");
            }

            if (target.shape()[0] !=
                prediction.shape()[0])
            {
                throw std::runtime_error(
                    "categorical_cross_entropy batch size mismatch");
            }

            if (valid_sample_count ==
                    0 ||
                valid_sample_count >
                    prediction.shape()[0])
            {
                throw std::runtime_error(
                    "categorical_cross_entropy valid sample count is out of range");
            }

            if (prediction.storage() !=
                    Storage::RowMajor ||
                target.storage() !=
                    Storage::RowMajor ||
                result.storage() !=
                    Storage::RowMajor)
            {
                throw std::runtime_error(
                    "categorical_cross_entropy currently supports only RowMajor tensors");
            }

            if (result.rank() !=
                    1 ||
                result.shape()[0] !=
                    1)
            {
                throw std::runtime_error(
                    "categorical_cross_entropy result must have shape {1}");
            }

            switch (reduction)
            {
            case Reduction::Mean:
            case Reduction::Sum:
                return;

            default:
                throw std::runtime_error(
                    "unknown Reduction in categorical_cross_entropy");
            }
        }

    }

    void categorical_cross_entropy(
        const Tensor &prediction,
        const Tensor &target,
        Tensor &result,
        Reduction reduction,
        std::size_t valid_sample_count)
    {
        validate_categorical_cross_entropy_inputs(
            prediction,
            target,
            result,
            reduction,
            valid_sample_count);

        switch (prediction.device().type())
        {
        case DeviceType::CPU:
            categorical_cross_entropy_cpu_float32(
                prediction,
                target,
                result,
                reduction,
                valid_sample_count);
            return;

        case DeviceType::CUDA:
#if defined(KL_ENABLE_CUDA)
            categorical_cross_entropy_cuda_float32(
                prediction,
                target,
                result,
                reduction,
                valid_sample_count);
            return;
#else
            throw std::runtime_error(
                "CUDA categorical_cross_entropy requested but CUDA backend is not enabled");
#endif

        case DeviceType::ROCM:
#if defined(KL_ENABLE_ROCM)
            categorical_cross_entropy_rocm_float32(
                prediction,
                target,
                result,
                reduction,
                valid_sample_count);
            return;
#else
            throw std::runtime_error(
                "ROCm categorical_cross_entropy requested but ROCm backend is not enabled");
#endif

        default:
            throw std::runtime_error(
                "unknown DeviceType in categorical_cross_entropy");
        }
    }

}