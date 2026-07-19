#include <gtest/gtest.h>

#ifdef KL_ENABLE_ROCM

#include "common/benchmark_report.hpp"
#include "common/benchmark_timer.hpp"
#include "common/tensor_compare.hpp"
#include "common/tensor_factory.hpp"

#include "vendor/rocm/rocblas_linear.hiph"

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>

#include <kernels/rocm/linear/linear_rocm_float32.hiph>

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

TEST(LinearForwardRocmFloat32, MatchesRocBLAS)
{
    const auto input = kl::test::makeRandomTensor(
        kl::Shape{batch_size, input_features},
        kl::DType::Float32,
        kl::Device::rocm(),
        -1.0,
        1.0,
        11);

    const auto weights = kl::test::makeRandomTensor(
        kl::Shape{output_features, input_features},
        kl::DType::Float32,
        kl::Device::rocm(),
        -1.0,
        1.0,
        22);

    kl::Tensor actual(
        kl::Shape{batch_size, output_features},
        kl::DType::Float32,
        kl::Device::rocm());

    kl::linear_rocm_float32(
        input,
        weights,
        nullptr,
        actual);

    const auto expected =
        kl::test::rocblasLinearForwardFloat32(input, weights);

    EXPECT_TRUE(kl::test::tensorCompare(
        expected,
        actual,
        absolute_tolerance,
        relative_tolerance));
}

TEST(LinearForwardRocmFloat32, BenchmarkAgainstRocBLAS)
{
    const auto input = kl::test::makeRandomTensor(
        kl::Shape{batch_size, input_features},
        kl::DType::Float32,
        kl::Device::rocm(),
        -1.0,
        1.0,
        11);

    const auto weights = kl::test::makeRandomTensor(
        kl::Shape{output_features, input_features},
        kl::DType::Float32,
        kl::Device::rocm(),
        -1.0,
        1.0,
        22);

    kl::Tensor output(
        kl::Shape{batch_size, output_features},
        kl::DType::Float32,
        kl::Device::rocm());

    const double kerneloom_ms = kl::test::benchmarkGpu(
        kl::Device::rocm(),
        warmup_iterations,
        measured_iterations,
        [&]()
        {
            kl::linear_rocm_float32(
                input,
                weights,
                nullptr,
                output);
        });

    const double rocblas_ms = kl::test::benchmarkGpu(
        kl::Device::rocm(),
        warmup_iterations,
        measured_iterations,
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