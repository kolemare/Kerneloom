#include <gtest/gtest.h>

#ifdef KL_ENABLE_CUDA

#include "common/benchmark_report.hpp"
#include "common/benchmark_timer.hpp"
#include "common/tensor_compare.hpp"
#include "common/tensor_factory.hpp"
#include "common/test_options.hpp"

#include "vendor/cuda/cublas_linear.cuh"

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>

#include <kernels/cuda/linear/linear_cuda_float32.cuh>

#include <cstddef>

namespace
{

    namespace options =
        kl::test::options::linear_forward_float32;

    void runCorrectnessNoBias(
        std::size_t batch_size,
        std::size_t input_features,
        std::size_t output_features)
    {
        const auto input = kl::test::makeRandomTensor(
            kl::Shape{
                batch_size,
                input_features},
            kl::DType::Float32,
            kl::Device::cuda(),
            -1.0,
            1.0,
            11);

        const auto weights = kl::test::makeRandomTensor(
            kl::Shape{
                output_features,
                input_features},
            kl::DType::Float32,
            kl::Device::cuda(),
            -1.0,
            1.0,
            22);

        kl::Tensor actual(
            kl::Shape{
                batch_size,
                output_features},
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
            options::absolute_tolerance,
            options::relative_tolerance));
    }

    void runCorrectnessWithBias(
        std::size_t batch_size,
        std::size_t input_features,
        std::size_t output_features)
    {
        const auto input = kl::test::makeRandomTensor(
            kl::Shape{
                batch_size,
                input_features},
            kl::DType::Float32,
            kl::Device::cuda(),
            -1.0,
            1.0,
            11);

        const auto weights = kl::test::makeRandomTensor(
            kl::Shape{
                output_features,
                input_features},
            kl::DType::Float32,
            kl::Device::cuda(),
            -1.0,
            1.0,
            22);

        const auto bias = kl::test::makeRandomTensor(
            kl::Shape{
                output_features},
            kl::DType::Float32,
            kl::Device::cuda(),
            -1.0,
            1.0,
            33);

        kl::Tensor actual(
            kl::Shape{
                batch_size,
                output_features},
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
            options::absolute_tolerance,
            options::relative_tolerance));
    }

    void runBenchmarkNoBias(
        const char *benchmark_name,
        std::size_t batch_size,
        std::size_t input_features,
        std::size_t output_features)
    {
        const auto input = kl::test::makeRandomTensor(
            kl::Shape{
                batch_size,
                input_features},
            kl::DType::Float32,
            kl::Device::cuda(),
            -1.0,
            1.0,
            11);

        const auto weights = kl::test::makeRandomTensor(
            kl::Shape{
                output_features,
                input_features},
            kl::DType::Float32,
            kl::Device::cuda(),
            -1.0,
            1.0,
            22);

        kl::Tensor kerneloom_output(
            kl::Shape{
                batch_size,
                output_features},
            kl::DType::Float32,
            kl::Device::cuda());

        kl::Tensor cublas_output(
            kl::Shape{
                batch_size,
                output_features},
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
            options::absolute_tolerance,
            options::relative_tolerance));

        kl::test::printBenchmarkComparison(
            benchmark_name,
            "Kerneloom CUDA",
            kerneloom_ms,
            "cuBLAS",
            cublas_ms);
    }

    void runBenchmarkWithBias(
        const char *benchmark_name,
        std::size_t batch_size,
        std::size_t input_features,
        std::size_t output_features)
    {
        const auto input = kl::test::makeRandomTensor(
            kl::Shape{
                batch_size,
                input_features},
            kl::DType::Float32,
            kl::Device::cuda(),
            -1.0,
            1.0,
            11);

        const auto weights = kl::test::makeRandomTensor(
            kl::Shape{
                output_features,
                input_features},
            kl::DType::Float32,
            kl::Device::cuda(),
            -1.0,
            1.0,
            22);

        const auto bias = kl::test::makeRandomTensor(
            kl::Shape{
                output_features},
            kl::DType::Float32,
            kl::Device::cuda(),
            -1.0,
            1.0,
            33);

        kl::Tensor kerneloom_output(
            kl::Shape{
                batch_size,
                output_features},
            kl::DType::Float32,
            kl::Device::cuda());

        kl::Tensor cublas_output(
            kl::Shape{
                batch_size,
                output_features},
            kl::DType::Float32,
            kl::Device::cuda());

        const auto ones =
            kl::test::makeOnesCudaFloat32(
                batch_size);

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
            options::absolute_tolerance,
            options::relative_tolerance));

        kl::test::printBenchmarkComparison(
            benchmark_name,
            "Kerneloom CUDA",
            kerneloom_ms,
            "cuBLAS",
            cublas_ms);
    }

}

TEST(LinearForwardCudaFloat32, MatchesCuBLAS_NoBias_LargeNonSquare)
{
    runCorrectnessNoBias(
        options::large_non_square::batch_size,
        options::large_non_square::input_features,
        options::large_non_square::output_features);
}

TEST(LinearForwardCudaFloat32, MatchesCuBLAS_WithBias_LargeNonSquare)
{
    runCorrectnessWithBias(
        options::large_non_square::batch_size,
        options::large_non_square::input_features,
        options::large_non_square::output_features);
}

TEST(LinearForwardCudaFloat32, MatchesCuBLAS_NoBias_DemandingOddShape)
{
    runCorrectnessNoBias(
        options::demanding_odd_shape::batch_size,
        options::demanding_odd_shape::input_features,
        options::demanding_odd_shape::output_features);
}

TEST(LinearForwardCudaFloat32, MatchesCuBLAS_WithBias_DemandingOddShape)
{
    runCorrectnessWithBias(
        options::demanding_odd_shape::batch_size,
        options::demanding_odd_shape::input_features,
        options::demanding_odd_shape::output_features);
}

TEST(LinearForwardCudaFloat32, BenchmarkAgainstCuBLAS_NoBias_LargeNonSquare)
{
    runBenchmarkNoBias(
        "Linear Forward CUDA Float32 No Bias Large Non-Square",
        options::large_non_square::batch_size,
        options::large_non_square::input_features,
        options::large_non_square::output_features);
}

TEST(LinearForwardCudaFloat32, BenchmarkAgainstCuBLAS_WithBias_LargeNonSquare)
{
    runBenchmarkWithBias(
        "Linear Forward CUDA Float32 With Bias Large Non-Square",
        options::large_non_square::batch_size,
        options::large_non_square::input_features,
        options::large_non_square::output_features);
}

TEST(LinearForwardCudaFloat32, BenchmarkAgainstCuBLAS_NoBias_DemandingOddShape)
{
    runBenchmarkNoBias(
        "Linear Forward CUDA Float32 No Bias Demanding Odd Shape",
        options::demanding_odd_shape::batch_size,
        options::demanding_odd_shape::input_features,
        options::demanding_odd_shape::output_features);
}

TEST(LinearForwardCudaFloat32, BenchmarkAgainstCuBLAS_WithBias_DemandingOddShape)
{
    runBenchmarkWithBias(
        "Linear Forward CUDA Float32 With Bias Demanding Odd Shape",
        options::demanding_odd_shape::batch_size,
        options::demanding_odd_shape::input_features,
        options::demanding_odd_shape::output_features);
}

#endif // KL_ENABLE_CUDA