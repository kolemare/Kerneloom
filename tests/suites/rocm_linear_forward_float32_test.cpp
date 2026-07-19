#include <gtest/gtest.h>

#ifdef KL_ENABLE_ROCM

#include "common/benchmark_report.hpp"
#include "common/benchmark_timer.hpp"
#include "common/tensor_compare.hpp"
#include "common/tensor_factory.hpp"
#include "common/test_options.hpp"

#include "vendor/rocm/rocblas_linear.hiph"

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>

#include <kernels/rocm/linear/linear_rocm_float32.hiph>

namespace
{

    namespace options =
        kl::test::options::linear_forward_float32;

}

TEST(LinearForwardRocmFloat32, MatchesRocBLAS)
{
    const auto input = kl::test::makeRandomTensor(
        kl::Shape{
            options::batch_size,
            options::input_features},
        kl::DType::Float32,
        kl::Device::rocm(),
        -1.0,
        1.0,
        11);

    const auto weights = kl::test::makeRandomTensor(
        kl::Shape{
            options::output_features,
            options::input_features},
        kl::DType::Float32,
        kl::Device::rocm(),
        -1.0,
        1.0,
        22);

    kl::Tensor actual(
        kl::Shape{
            options::batch_size,
            options::output_features},
        kl::DType::Float32,
        kl::Device::rocm());

    kl::linear_rocm_float32(
        input,
        weights,
        nullptr,
        actual);

    const auto expected =
        kl::test::rocblasLinearForwardFloat32(
            input,
            weights);

    EXPECT_TRUE(kl::test::tensorCompare(
        expected,
        actual,
        options::absolute_tolerance,
        options::relative_tolerance));
}

TEST(LinearForwardRocmFloat32, BenchmarkAgainstRocBLAS)
{
    const auto input = kl::test::makeRandomTensor(
        kl::Shape{
            options::batch_size,
            options::input_features},
        kl::DType::Float32,
        kl::Device::rocm(),
        -1.0,
        1.0,
        11);

    const auto weights = kl::test::makeRandomTensor(
        kl::Shape{
            options::output_features,
            options::input_features},
        kl::DType::Float32,
        kl::Device::rocm(),
        -1.0,
        1.0,
        22);

    kl::Tensor kerneloom_output(
        kl::Shape{
            options::batch_size,
            options::output_features},
        kl::DType::Float32,
        kl::Device::rocm());

    const double kerneloom_ms = kl::test::benchmarkGpu(
        kl::Device::rocm(),
        options::warmup_iterations,
        options::measured_iterations,
        options::print_each_iteration,
        "Kerneloom ROCm",
        [&]()
        {
            kl::linear_rocm_float32(
                input,
                weights,
                nullptr,
                kerneloom_output);
        });

    const double rocblas_ms = kl::test::benchmarkGpu(
        kl::Device::rocm(),
        options::warmup_iterations,
        options::measured_iterations,
        options::print_each_iteration,
        "rocBLAS",
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