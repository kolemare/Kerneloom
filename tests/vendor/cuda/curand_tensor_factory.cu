#include "vendor/cuda/curand_tensor_factory.cuh"

#ifdef KL_ENABLE_CUDA

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/tensor.hpp>

#include <cuda_runtime.h>
#include <curand.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace kl::test
{

    namespace
    {

        void check_cuda(
            const cudaError_t error,
            const char *message)
        {
            if (error != cudaSuccess)
            {
                throw std::runtime_error(
                    std::string(message) + ": " +
                    cudaGetErrorString(error));
            }
        }

        void check_curand(
            const curandStatus_t status,
            const char *message)
        {
            if (status != CURAND_STATUS_SUCCESS)
            {
                throw std::runtime_error(
                    message);
            }
        }

        __global__ void scale_random_cuda_float32_kernel(
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

        void scaleRandomCudaFloat32(
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

            scale_random_cuda_float32_kernel<<<grid, block>>>(
                static_cast<float *>(tensor.data()),
                size,
                static_cast<float>(min_value),
                static_cast<float>(max_value - min_value));

            check_cuda(
                cudaGetLastError(),
                "CUDA random tensor scale kernel launch failed");
        }

        class CurandGenerator
        {

        public:
            explicit CurandGenerator(
                const unsigned long long seed)
            {
                check_curand(
                    curandCreateGenerator(
                        &generator_,
                        CURAND_RNG_PSEUDO_DEFAULT),
                    "cuRAND generator creation failed");

                check_curand(
                    curandSetPseudoRandomGeneratorSeed(
                        generator_,
                        seed),
                    "cuRAND seed setup failed");
            }

            ~CurandGenerator()
            {
                if (generator_ != nullptr)
                {
                    curandDestroyGenerator(
                        generator_);
                }
            }

            CurandGenerator(
                const CurandGenerator &) = delete;

            CurandGenerator &operator=(
                const CurandGenerator &) = delete;

            curandGenerator_t get() const
            {
                return generator_;
            }

        private:
            curandGenerator_t generator_ =
                nullptr;
        };

    }

    Tensor makeRandomCudaTensorFloat32(
        Shape shape,
        double min_value,
        double max_value,
        unsigned int seed)
    {
        Tensor tensor(
            std::move(shape),
            DType::Float32,
            Device::cuda());

        CurandGenerator generator(
            static_cast<unsigned long long>(seed));

        check_curand(
            curandGenerateUniform(
                generator.get(),
                static_cast<float *>(tensor.data()),
                tensor.numel()),
            "cuRAND uniform generation failed");

        scaleRandomCudaFloat32(
            tensor,
            min_value,
            max_value);

        return tensor;
    }

}

#endif // KL_ENABLE_CUDA