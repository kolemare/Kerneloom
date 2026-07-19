#include <gtest/gtest.h>

#include "common/benchmark_report.hpp"
#include "common/benchmark_timer.hpp"
#include "common/tensor_compare.hpp"
#include "common/tensor_factory.hpp"

#ifdef KL_ENABLE_CUDA
#include "vendor/cuda/cublas_linear.cuh"
#include <kernels/cuda/linear/linear_cuda_float32.cuh>
#endif

#ifdef KL_ENABLE_ROCM
#include "vendor/rocm/rocblas_linear.hiph"
#include <kernels/rocm/linear/linear_rocm_float32.hiph>
#endif

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/synchronize.hpp>
#include <core/tensor.hpp>

#include <cstddef>

namespace
{

    struct LinearForwardFloat32Config
    {
        std::size_t batch_size;
        std::size_t input_features;
        std::size_t output_features;

        double absolute_tolerance;
        double relative_tolerance;

        std::size_t warmup_iterations;
        std::size_t measured_iterations;
    };

    constexpr LinearForwardFloat32Config config{
        .batch_size = 2048,
        .input_features = 8192,
        .output_features = 8192,
        .absolute_tolerance = 1.0e-3,
        .relative_tolerance = 1.0e-3,
        .warmup_iterations = 3,
        .measured_iterations = 5};

    kl::Tensor makeInput(
        kl::Device device)
    {
        return kl::test::makeRandomTensor(
            kl::Shape{config.batch_size, config.input_features},
            kl::DType::Float32,
            device,
            -1.0,
            1.0,
            11);
    }

    kl::Tensor makeWeights(
        kl::Device device)
    {
        return kl::test::makeRandomTensor(
            kl::Shape{config.output_features, config.input_features},
            kl::DType::Float32,
            device,
            -1.0,
            1.0,
            22);
    }

#ifdef KL_ENABLE_CUDA

    kl::Tensor kerneloomLinearCudaFloat32(
        const kl::Tensor &input,
        const kl::Tensor &weights)
    {
        kl::Tensor output(
            kl::Shape{input.shape()[0], weights.shape()[0]},
            kl::DType::Float32,
            kl::Device::cuda());

        kl::linear_cuda_float32(
            input,
            weights,
            nullptr,
            output);

        return output;
    }

#endif // KL_ENABLE_CUDA

#ifdef KL_ENABLE_ROCM

    kl::Tensor kerneloomLinearRocmFloat32(
        const kl::Tensor &input,
        const kl::Tensor &weights)
    {
        kl::Tensor output(
            kl::Shape{input.shape()[0], weights.shape()[0]},
            kl::DType::Float32,
            kl::Device::rocm());

        kl::linear_rocm_float32(
            input,
            weights,
            nullptr,
            output);

        return output;
    }

#endif // KL_ENABLE_ROCM

    template <typename Fn>
    double benchmarkGpu(
        kl::Device device,
        Fn &&fn)
    {
        for (std::size_t i = 0; i < config.warmup_iterations; ++i)
        {
            (void)fn();
        }

        kl::synchronize(device);

        double total_ms = 0.0;

        for (std::size_t i = 0; i < config.measured_iterations; ++i)
        {
            kl::synchronize(device);

            kl::test::BenchmarkTimer timer;
            timer.start();

            (void)fn();

            kl::synchronize(device);

            total_ms += timer.stopMilliseconds();
        }

        return total_ms / static_cast<double>(config.measured_iterations);
    }

}

#ifdef KL_ENABLE_CUDA

TEST(LinearForwardCudaFloat32, MatchesCuBLAS)
{
    const auto input =
        makeInput(kl::Device::cuda());

    const auto weights =
        makeWeights(kl::Device::cuda());

    const auto expected =
        kl::test::cublasLinearForwardFloat32(input, weights);

    const auto actual =
        kerneloomLinearCudaFloat32(input, weights);

    EXPECT_TRUE(kl::test::tensorCompare(
        expected,
        actual,
        config.absolute_tolerance,
        config.relative_tolerance));
}

TEST(LinearForwardCudaFloat32, BenchmarkAgainstCuBLAS)
{
    const auto input =
        makeInput(kl::Device::cuda());

    const auto weights =
        makeWeights(kl::Device::cuda());

    const double kerneloom_ms = benchmarkGpu(
        kl::Device::cuda(),
        [&]()
        {
            return kerneloomLinearCudaFloat32(
                input,
                weights);
        });

    const double cublas_ms = benchmarkGpu(
        kl::Device::cuda(),
        [&]()
        {
            return kl::test::cublasLinearForwardFloat32(
                input,
                weights);
        });

    kl::test::printBenchmarkComparison(
        "Linear Forward CUDA Float32",
        "Kerneloom CUDA",
        kerneloom_ms,
        "cuBLAS",
        cublas_ms);
}

#endif // KL_ENABLE_CUDA

#ifdef KL_ENABLE_ROCM

TEST(LinearForwardRocmFloat32, MatchesRocBLAS)
{
    const auto input =
        makeInput(kl::Device::rocm());

    const auto weights =
        makeWeights(kl::Device::rocm());

    const auto expected =
        kl::test::rocblasLinearForwardFloat32(input, weights);

    const auto actual =
        kerneloomLinearRocmFloat32(input, weights);

    EXPECT_TRUE(kl::test::tensorCompare(
        expected,
        actual,
        config.absolute_tolerance,
        config.relative_tolerance));
}

TEST(LinearForwardRocmFloat32, BenchmarkAgainstRocBLAS)
{
    const auto input =
        makeInput(kl::Device::rocm());

    const auto weights =
        makeWeights(kl::Device::rocm());

    const double kerneloom_ms = benchmarkGpu(
        kl::Device::rocm(),
        [&]()
        {
            return kerneloomLinearRocmFloat32(
                input,
                weights);
        });

    const double rocblas_ms = benchmarkGpu(
        kl::Device::rocm(),
        [&]()
        {
            return kl::test::rocblasLinearForwardFloat32(
                input,
                weights);
        });

    kl::test::printBenchmarkComparison(
        "Linear Forward ROCm Float32",
        "Kerneloom ROCm",
        kerneloom_ms,
        "rocBLAS",
        rocblas_ms);
}

#endif // KL_ENABLE_ROCM