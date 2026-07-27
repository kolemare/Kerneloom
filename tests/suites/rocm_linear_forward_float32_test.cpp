#include <gtest/gtest.h>

#ifdef KL_ENABLE_ROCM

#include "common/benchmark_report.hpp"
#include "common/benchmark_timer.hpp"
#include "common/tensor_compare.hpp"
#include "common/tensor_factory.hpp"
#include "common/test_options.hpp"

#include "options/linear_forward_options.hpp"

#include "vendor/rocm/rocblas_linear.hiph"

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>

#include <kernels/rocm/linear/linear_rocm_float32.hiph>

#include <cstddef>
#include <string>

namespace
{

    namespace options =
        kl::test::options::linear_forward;

    void runCorrectnessNoBias(
        const options::Shape &shape)
    {
        SCOPED_TRACE(
            std::string("Shape: ") +
            shape.name);

        const auto input = kl::test::makeRandomTensor(
            kl::Shape{
                shape.batch_size,
                shape.input_features},
            kl::DType::Float32,
            kl::Device::rocm(),
            -1.0,
            1.0,
            11);

        const auto weights = kl::test::makeRandomTensor(
            kl::Shape{
                shape.output_features,
                shape.input_features},
            kl::DType::Float32,
            kl::Device::rocm(),
            -1.0,
            1.0,
            22);

        kl::Tensor actual(
            kl::Shape{
                shape.batch_size,
                shape.output_features},
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
            options::float32_absolute_tolerance,
            options::float32_relative_tolerance,
            kl::test::options::tensor_compare::max_mismatch_ratio));
    }

    void runCorrectnessWithBias(
        const options::Shape &shape)
    {
        SCOPED_TRACE(
            std::string("Shape: ") +
            shape.name);

        const auto input = kl::test::makeRandomTensor(
            kl::Shape{
                shape.batch_size,
                shape.input_features},
            kl::DType::Float32,
            kl::Device::rocm(),
            -1.0,
            1.0,
            11);

        const auto weights = kl::test::makeRandomTensor(
            kl::Shape{
                shape.output_features,
                shape.input_features},
            kl::DType::Float32,
            kl::Device::rocm(),
            -1.0,
            1.0,
            22);

        const auto bias = kl::test::makeRandomTensor(
            kl::Shape{
                shape.output_features},
            kl::DType::Float32,
            kl::Device::rocm(),
            -1.0,
            1.0,
            33);

        kl::Tensor actual(
            kl::Shape{
                shape.batch_size,
                shape.output_features},
            kl::DType::Float32,
            kl::Device::rocm());

        kl::linear_rocm_float32(
            input,
            weights,
            &bias,
            actual);

        const auto expected =
            kl::test::rocblasLinearForwardFloat32(
                input,
                weights,
                bias);

        EXPECT_TRUE(kl::test::tensorCompare(
            expected,
            actual,
            options::float32_absolute_tolerance,
            options::float32_relative_tolerance,
            kl::test::options::tensor_compare::max_mismatch_ratio));
    }

    void runBenchmarkNoBias(
        const options::Shape &shape)
    {
        const auto input = kl::test::makeRandomTensor(
            kl::Shape{
                shape.batch_size,
                shape.input_features},
            kl::DType::Float32,
            kl::Device::rocm(),
            -1.0,
            1.0,
            11);

        const auto weights = kl::test::makeRandomTensor(
            kl::Shape{
                shape.output_features,
                shape.input_features},
            kl::DType::Float32,
            kl::Device::rocm(),
            -1.0,
            1.0,
            22);

        kl::Tensor kerneloom_output(
            kl::Shape{
                shape.batch_size,
                shape.output_features},
            kl::DType::Float32,
            kl::Device::rocm());

        kl::Tensor rocblas_output(
            kl::Shape{
                shape.batch_size,
                shape.output_features},
            kl::DType::Float32,
            kl::Device::rocm());

        kl::test::RocblasHandle rocblas_handle;

        const double kerneloom_ms = kl::test::benchmarkGpu(
            kl::Device::rocm(),
            options::warmup_iterations,
            options::measured_iterations,
            kl::test::options::benchmark::print_each_iteration,
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
            kl::test::options::benchmark::print_each_iteration,
            "rocBLAS",
            [&]()
            {
                kl::test::rocblasLinearForwardFloat32(
                    rocblas_handle,
                    input,
                    weights,
                    rocblas_output);
            });

        EXPECT_TRUE(kl::test::tensorCompare(
            rocblas_output,
            kerneloom_output,
            options::float32_absolute_tolerance,
            options::float32_relative_tolerance,
            kl::test::options::tensor_compare::max_mismatch_ratio));

        const std::string benchmark_name =
            std::string("Linear Forward ROCm Float32 No Bias ") +
            shape.name;

        kl::test::printBenchmarkComparison(
            benchmark_name.c_str(),
            "Kerneloom ROCm",
            kerneloom_ms,
            "rocBLAS",
            rocblas_ms);
    }

    void runBenchmarkWithBias(
        const options::Shape &shape)
    {
        const auto input = kl::test::makeRandomTensor(
            kl::Shape{
                shape.batch_size,
                shape.input_features},
            kl::DType::Float32,
            kl::Device::rocm(),
            -1.0,
            1.0,
            11);

        const auto weights = kl::test::makeRandomTensor(
            kl::Shape{
                shape.output_features,
                shape.input_features},
            kl::DType::Float32,
            kl::Device::rocm(),
            -1.0,
            1.0,
            22);

        const auto bias = kl::test::makeRandomTensor(
            kl::Shape{
                shape.output_features},
            kl::DType::Float32,
            kl::Device::rocm(),
            -1.0,
            1.0,
            33);

        kl::Tensor kerneloom_output(
            kl::Shape{
                shape.batch_size,
                shape.output_features},
            kl::DType::Float32,
            kl::Device::rocm());

        kl::Tensor rocblas_output(
            kl::Shape{
                shape.batch_size,
                shape.output_features},
            kl::DType::Float32,
            kl::Device::rocm());

        const auto ones =
            kl::test::makeOnesRocmFloat32(
                shape.batch_size);

        kl::test::RocblasHandle rocblas_handle;

        const double kerneloom_ms = kl::test::benchmarkGpu(
            kl::Device::rocm(),
            options::warmup_iterations,
            options::measured_iterations,
            kl::test::options::benchmark::print_each_iteration,
            "Kerneloom ROCm",
            [&]()
            {
                kl::linear_rocm_float32(
                    input,
                    weights,
                    &bias,
                    kerneloom_output);
            });

        const double rocblas_ms = kl::test::benchmarkGpu(
            kl::Device::rocm(),
            options::warmup_iterations,
            options::measured_iterations,
            kl::test::options::benchmark::print_each_iteration,
            "rocBLAS",
            [&]()
            {
                kl::test::rocblasLinearForwardFloat32(
                    rocblas_handle,
                    input,
                    weights,
                    bias,
                    ones,
                    rocblas_output);
            });

        EXPECT_TRUE(kl::test::tensorCompare(
            rocblas_output,
            kerneloom_output,
            options::float32_absolute_tolerance,
            options::float32_relative_tolerance,
            kl::test::options::tensor_compare::max_mismatch_ratio));

        const std::string benchmark_name =
            std::string("Linear Forward ROCm Float32 With Bias ") +
            shape.name;

        kl::test::printBenchmarkComparison(
            benchmark_name.c_str(),
            "Kerneloom ROCm",
            kerneloom_ms,
            "rocBLAS",
            rocblas_ms);
    }

}

TEST(LinearForwardRocmFloat32, MatchesRocBLAS_NoBias)
{
    for (const auto &shape : options::correctness_shapes)
    {
        runCorrectnessNoBias(
            shape);
    }
}

TEST(LinearForwardRocmFloat32, MatchesRocBLAS_WithBias)
{
    for (const auto &shape : options::correctness_shapes)
    {
        runCorrectnessWithBias(
            shape);
    }
}

TEST(LinearForwardRocmFloat32, BenchmarkAgainstRocBLAS_NoBias)
{
    for (const auto &shape : options::no_bias_benchmark_shapes)
    {
        runBenchmarkNoBias(
            shape);
    }
}

TEST(LinearForwardRocmFloat32, BenchmarkAgainstRocBLAS_WithBias)
{
    for (const auto &shape : options::bias_benchmark_shapes)
    {
        runBenchmarkWithBias(
            shape);
    }
}

#endif // KL_ENABLE_ROCM