#include <gtest/gtest.h>

#ifdef KL_ENABLE_CUDA

#include "common/benchmark_report.hpp"
#include "common/benchmark_timer.hpp"
#include "common/tensor_compare.hpp"
#include "common/tensor_factory.hpp"

#include "vendor/cuda/cublas_linear.cuh"

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>

#include <kernels/cuda/linear/linear_cuda_float32.cuh>

#include <cstddef>

namespace
{

    constexpr std::size_t batch_size = 2048;
    constexpr std::size_t input_features = 8192;
    constexpr std::size_t output_features = 8192;

    constexpr double absolute_tolerance = 1.0e-3;
    constexpr double relative_tolerance = 1.0e-3;

    constexpr std::size_t warmup_iterations = 3;
    constexpr std::size_t measured_iterations = 5;

}

TEST(LinearForwardCudaFloat32, MatchesCuBLAS)
{
    const auto input = kl::test::makeRandomTensor(
        kl::Shape{batch_size, input_features},
        kl::DType::Float32,
        kl::Device::cuda(),
        -1.0,
        1.0,
        11);

    const auto weights = kl::test::makeRandomTensor(
        kl::Shape{output_features, input_features},
        kl::DType::Float32,
        kl::Device::cuda(),
        -1.0,
        1.0,
        22);

    kl::Tensor actual(
        kl::Shape{batch_size, output_features},
        kl::DType::Float32,
        kl::Device::cuda());

    kl::linear_cuda_float32(
        input,
        weights,
        nullptr,
        actual);

    const auto expected =
        kl::test::cublasLinearForwardFloat32(input, weights);

    EXPECT_TRUE(kl::test::tensorCompare(
        expected,
        actual,
        absolute_tolerance,
        relative_tolerance));
}

TEST(LinearForwardCudaFloat32, BenchmarkAgainstCuBLAS)
{
    const auto input = kl::test::makeRandomTensor(
        kl::Shape{batch_size, input_features},
        kl::DType::Float32,
        kl::Device::cuda(),
        -1.0,
        1.0,
        11);

    const auto weights = kl::test::makeRandomTensor(
        kl::Shape{output_features, input_features},
        kl::DType::Float32,
        kl::Device::cuda(),
        -1.0,
        1.0,
        22);

    kl::Tensor output(
        kl::Shape{batch_size, output_features},
        kl::DType::Float32,
        kl::Device::cuda());

    const double kerneloom_ms = kl::test::benchmarkGpu(
        kl::Device::cuda(),
        warmup_iterations,
        measured_iterations,
        [&]()
        {
            kl::linear_cuda_float32(
                input,
                weights,
                nullptr,
                output);
        });

    const double cublas_ms = kl::test::benchmarkGpu(
        kl::Device::cuda(),
        warmup_iterations,
        measured_iterations,
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