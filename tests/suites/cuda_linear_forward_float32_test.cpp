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

namespace
{

    namespace options =
        kl::test::options::linear_forward_float32;

}

TEST(LinearForwardCudaFloat32, MatchesCuBLAS)
{
    const auto input = kl::test::makeRandomTensor(
        kl::Shape{
            options::batch_size,
            options::input_features},
        kl::DType::Float32,
        kl::Device::cuda(),
        -1.0,
        1.0,
        11);

    const auto weights = kl::test::makeRandomTensor(
        kl::Shape{
            options::output_features,
            options::input_features},
        kl::DType::Float32,
        kl::Device::cuda(),
        -1.0,
        1.0,
        22);

    kl::Tensor actual(
        kl::Shape{
            options::batch_size,
            options::output_features},
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

TEST(LinearForwardCudaFloat32, BenchmarkAgainstCuBLAS)
{
    const auto input = kl::test::makeRandomTensor(
        kl::Shape{
            options::batch_size,
            options::input_features},
        kl::DType::Float32,
        kl::Device::cuda(),
        -1.0,
        1.0,
        11);

    const auto weights = kl::test::makeRandomTensor(
        kl::Shape{
            options::output_features,
            options::input_features},
        kl::DType::Float32,
        kl::Device::cuda(),
        -1.0,
        1.0,
        22);

    kl::Tensor kerneloom_output(
        kl::Shape{
            options::batch_size,
            options::output_features},
        kl::DType::Float32,
        kl::Device::cuda());

    kl::Tensor cublas_output(
        kl::Shape{
            options::batch_size,
            options::output_features},
        kl::DType::Float32,
        kl::Device::cuda());

    kl::test::CublasHandle cublas_handle;

    const double kerneloom_ms = kl::test::benchmarkGpu(
        kl::Device::cuda(),
        options::warmup_iterations,
        options::measured_iterations,
        options::print_each_iteration,
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
        options::print_each_iteration,
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
        "Linear Forward CUDA Float32",
        "Kerneloom CUDA",
        kerneloom_ms,
        "cuBLAS",
        cublas_ms);
}

#endif // KL_ENABLE_CUDA