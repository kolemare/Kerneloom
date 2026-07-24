#include "vendor/rocm/hiprand_tensor_factory.hiph"

#ifdef KL_ENABLE_ROCM

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/tensor.hpp>

#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace kl::test
{

    namespace
    {

        void check_hip(
            const hipError_t error,
            const char *message)
        {
            if (error != hipSuccess)
            {
                throw std::runtime_error(
                    std::string(message) + ": " +
                    hipGetErrorString(error));
            }
        }

        void check_hiprand(
            const hiprandStatus_t status,
            const char *message)
        {
            if (status != HIPRAND_STATUS_SUCCESS)
            {
                throw std::runtime_error(
                    message);
            }
        }

        __global__ void scale_random_rocm_float32_kernel(
            float *__restrict__ data,
            const std::size_t size,
            const float min_value,
            const float range)
        {
            const std::size_t index =
                static_cast<std::size_t>(blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            if (index >= size)
            {
                return;
            }

            data[index] =
                min_value +
                data[index] * range;
        }

        void scaleRandomRocmFloat32(
            Tensor &tensor,
            const double min_value,
            const double max_value)
        {
            constexpr int block_size =
                256;

            const std::size_t size =
                tensor.numel();

            const dim3 block(
                block_size);

            const dim3 grid(
                static_cast<unsigned int>(
                    (size + block_size - 1) /
                    block_size));

            hipLaunchKernelGGL(
                scale_random_rocm_float32_kernel,
                grid,
                block,
                0,
                0,
                static_cast<float *>(tensor.data()),
                size,
                static_cast<float>(min_value),
                static_cast<float>(max_value - min_value));

            check_hip(
                hipGetLastError(),
                "ROCm random tensor scale kernel launch failed");
        }

        class HiprandGenerator
        {

        public:
            explicit HiprandGenerator(
                const unsigned long long seed)
            {
                check_hiprand(
                    hiprandCreateGenerator(
                        &generator_,
                        HIPRAND_RNG_PSEUDO_DEFAULT),
                    "hipRAND generator creation failed");

                check_hiprand(
                    hiprandSetPseudoRandomGeneratorSeed(
                        generator_,
                        seed),
                    "hipRAND seed setup failed");
            }

            ~HiprandGenerator()
            {
                if (generator_ != nullptr)
                {
                    hiprandDestroyGenerator(
                        generator_);
                }
            }

            HiprandGenerator(
                const HiprandGenerator &) = delete;

            HiprandGenerator &operator=(
                const HiprandGenerator &) = delete;

            hiprandGenerator_t get() const
            {
                return generator_;
            }

        private:
            hiprandGenerator_t generator_ =
                nullptr;
        };

    }

    Tensor makeRandomRocmTensorFloat32(
        Shape shape,
        double min_value,
        double max_value,
        unsigned int seed)
    {
        Tensor tensor(
            std::move(shape),
            DType::Float32,
            Device::rocm());

        HiprandGenerator generator(
            static_cast<unsigned long long>(seed));

        check_hiprand(
            hiprandGenerateUniform(
                generator.get(),
                static_cast<float *>(tensor.data()),
                tensor.numel()),
            "hipRAND uniform generation failed");

        scaleRandomRocmFloat32(
            tensor,
            min_value,
            max_value);

        return tensor;
    }

}

#endif // KL_ENABLE_ROCM