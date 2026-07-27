#include <gtest/gtest.h>

#ifdef KL_ENABLE_CUDA

#include "common/benchmark_report.hpp"
#include "common/benchmark_timer.hpp"
#include "common/tensor_compare.hpp"
#include "common/tensor_factory.hpp"
#include "common/test_options.hpp"

#include "options/linear_forward_options.hpp"

#include "vendor/cuda/cublas_linear.cuh"

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>

#include <kernels/cuda/linear/linear_cuda_float32.cuh>

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
            kl::Device::cuda(),
            -1.0,
            1.0,
            11);

        const auto weights = kl::test::makeRandomTensor(
            kl::Shape{
                shape.output_features,
                shape.input_features},
            kl::DType::Float32,
            kl::Device::cuda(),
            -1.0,
            1.0,
            22);

        kl::Tensor actual(
            kl::Shape{
                shape.batch_size,
                shape.output_features},
            kl::DType::Float32,
            kl::Device::cuda());

        kl::linear_cuda_float32(
            input,
            weights,
            nullptr,
            actual);

        const auto expected =
            kl::test::cublasLinearForwardFloat32(
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
            kl::Device::cuda(),
            -1.0,
            1.0,
            11);

        const auto weights = kl::test::makeRandomTensor(
            kl::Shape{
                shape.output_features,
                shape.input_features},
            kl::DType::Float32,
            kl::Device::cuda(),
            -1.0,
            1.0,
            22);

        const auto bias = kl::test::makeRandomTensor(
            kl::Shape{
                shape.output_features},
            kl::DType::Float32,
            kl::Device::cuda(),
            -1.0,
            1.0,
            33);

        kl::Tensor actual(
            kl::Shape{
                shape.batch_size,
                shape.output_features},
            kl::DType::Float32,
            kl::Device::cuda());

        kl::linear_cuda_float32(
            input,
            weights,
            &bias,
            actual);

        const auto expected =
            kl::test::cublasLinearForwardFloat32(
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
            kl::Device::cuda(),
            -1.0,
            1.0,
            11);

        const auto weights = kl::test::makeRandomTensor(
            kl::Shape{
                shape.output_features,
                shape.input_features},
            kl::DType::Float32,
            kl::Device::cuda(),
            -1.0,
            1.0,
            22);

        kl::Tensor kerneloom_output(
            kl::Shape{
                shape.batch_size,
                shape.output_features},
            kl::DType::Float32,
            kl::Device::cuda());

        kl::Tensor cublas_output(
            kl::Shape{
                shape.batch_size,
                shape.output_features},
            kl::DType::Float32,
            kl::Device::cuda());

        kl::test::CublasHandle cublas_handle;

        const double kerneloom_ms = kl::test::benchmarkGpu(
            kl::Device::cuda(),
            options::warmup_iterations,
            options::measured_iterations,
            kl::test::options::benchmark::print_each_iteration,
            "Kerneloom CUDA",
            [&]()
            {
                kl::linear_cuda_float32(
                    input,
                    weights,
                    nullptr,
                    kerneloom_output);
            });

        const double cublas_ms = kl::test::benchmarkGpu(
            kl::Device::cuda(),
            options::warmup_iterations,
            options::measured_iterations,
            kl::test::options::benchmark::print_each_iteration,
            "cuBLAS",
            [&]()
            {
                kl::test::cublasLinearForwardFloat32(
                    cublas_handle,
                    input,
                    weights,
                    cublas_output);
            });

        EXPECT_TRUE(kl::test::tensorCompare(
            cublas_output,
            kerneloom_output,
            options::float32_absolute_tolerance,
            options::float32_relative_tolerance,
            kl::test::options::tensor_compare::max_mismatch_ratio));

        kl::test::printBenchmarkComparison(
            (std::string("Linear Forward CUDA Float32 No Bias ") +
             shape.name)
                .c_str(),
            "Kerneloom CUDA",
            kerneloom_ms,
            "cuBLAS",
            cublas_ms);
    }

    void runBenchmarkWithBias(
        const options::Shape &shape)
    {
        const auto input = kl::test::makeRandomTensor(
            kl::Shape{
                shape.batch_size,
                shape.input_features},
            kl::DType::Float32,
            kl::Device::cuda(),
            -1.0,
            1.0,
            11);

        const auto weights = kl::test::makeRandomTensor(
            kl::Shape{
                shape.output_features,
                shape.input_features},
            kl::DType::Float32,
            kl::Device::cuda(),
            -1.0,
            1.0,
            22);

        const auto bias = kl::test::makeRandomTensor(
            kl::Shape{
                shape.output_features},
            kl::DType::Float32,
            kl::Device::cuda(),
            -1.0,
            1.0,
            33);

        kl::Tensor kerneloom_output(
            kl::Shape{
                shape.batch_size,
                shape.output_features},
            kl::DType::Float32,
            kl::Device::cuda());

        kl::Tensor cublas_output(
            kl::Shape{
                shape.batch_size,
                shape.output_features},
            kl::DType::Float32,
            kl::Device::cuda());

        const auto ones =
            kl::test::makeOnesCudaFloat32(
                shape.batch_size);

        kl::test::CublasHandle cublas_handle;

        const double kerneloom_ms = kl::test::benchmarkGpu(
            kl::Device::cuda(),
            options::warmup_iterations,
            options::measured_iterations,
            kl::test::options::benchmark::print_each_iteration,
            "Kerneloom CUDA",
            [&]()
            {
                kl::linear_cuda_float32(
                    input,
                    weights,
                    &bias,
                    kerneloom_output);
            });

        const double cublas_ms = kl::test::benchmarkGpu(
            kl::Device::cuda(),
            options::warmup_iterations,
            options::measured_iterations,
            kl::test::options::benchmark::print_each_iteration,
            "cuBLAS",
            [&]()
            {
                kl::test::cublasLinearForwardFloat32(
                    cublas_handle,
                    input,
                    weights,
                    bias,
                    ones,
                    cublas_output);
            });

        EXPECT_TRUE(kl::test::tensorCompare(
            cublas_output,
            kerneloom_output,
            options::float32_absolute_tolerance,
            options::float32_relative_tolerance,
            kl::test::options::tensor_compare::max_mismatch_ratio));

        kl::test::printBenchmarkComparison(
            (std::string("Linear Forward CUDA Float32 With Bias ") +
             shape.name)
                .c_str(),
            "Kerneloom CUDA",
            kerneloom_ms,
            "cuBLAS",
            cublas_ms);
    }

}

TEST(LinearForwardCudaFloat32, MatchesCuBLAS_NoBias)
{
    for (const auto &shape : options::correctness_shapes)
    {
        runCorrectnessNoBias(
            shape);
    }
}

TEST(LinearForwardCudaFloat32, MatchesCuBLAS_WithBias)
{
    for (const auto &shape : options::correctness_shapes)
    {
        runCorrectnessWithBias(
            shape);
    }
}

TEST(LinearForwardCudaFloat32, BenchmarkAgainstCuBLAS_NoBias)
{
    for (const auto &shape : options::no_bias_benchmark_shapes)
    {
        runBenchmarkNoBias(
            shape);
    }
}

TEST(LinearForwardCudaFloat32, BenchmarkAgainstCuBLAS_WithBias)
{
    for (const auto &shape : options::bias_benchmark_shapes)
    {
        runBenchmarkWithBias(
            shape);
    }
}

#endif // KL_ENABLE_CUDA