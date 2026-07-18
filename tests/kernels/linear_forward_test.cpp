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

    constexpr std::size_t batch_size = 2048;
    constexpr std::size_t input_features = 8192;
    constexpr std::size_t output_features = 8192;

    constexpr std::size_t warmup_iterations = 3;
    constexpr std::size_t measured_iterations = 5;

    constexpr double float32_absolute_tolerance = 1.0e-3;
    constexpr double float32_relative_tolerance = 1.0e-3;

#ifdef KL_ENABLE_CUDA

    kl::Tensor kerneloomLinearForwardCudaFloat32(
        const kl::Tensor &input,
        const kl::Tensor &weights)
    {
        kl::Tensor result(
            kl::Shape{batch_size, output_features},
            kl::DType::Float32,
            kl::Device::cuda());

        kl::linear_cuda_float32(
            input,
            weights,
            nullptr,
            result);

        return result;
    }

#endif // KL_ENABLE_CUDA

#ifdef KL_ENABLE_ROCM

    kl::Tensor kerneloomLinearForwardRocmFloat32(
        const kl::Tensor &input,
        const kl::Tensor &weights)
    {
        kl::Tensor result(
            kl::Shape{batch_size, output_features},
            kl::DType::Float32,
            kl::Device::rocm());

        kl::linear_rocm_float32(
            input,
            weights,
            nullptr,
            result);

        return result;
    }

#endif // KL_ENABLE_ROCM

    template <typename Fn>
    double benchmarkGpu(
        kl::Device device,
        Fn &&fn)
    {
        for (std::size_t i = 0; i < warmup_iterations; ++i)
        {
            (void)fn();
        }

        kl::synchronize(device);

        double total_ms = 0.0;

        for (std::size_t i = 0; i < measured_iterations; ++i)
        {
            kl::synchronize(device);

            kl::test::BenchmarkTimer timer;
            timer.start();

            (void)fn();

            kl::synchronize(device);

            total_ms += timer.stopMilliseconds();
        }

        return total_ms / static_cast<double>(measured_iterations);
    }

}

#ifdef KL_ENABLE_CUDA

TEST(LinearForwardCudaFloat32, MatchesCuBLAS)
{
    const kl::Tensor input = kl::test::makeRandomTensor(
        kl::Shape{batch_size, input_features},
        kl::DType::Float32,
        kl::Device::cuda(),
        -1.0,
        1.0,
        11);

    const kl::Tensor weights = kl::test::makeRandomTensor(
        kl::Shape{output_features, input_features},
        kl::DType::Float32,
        kl::Device::cuda(),
        -1.0,
        1.0,
        22);

    const kl::Tensor expected = kl::test::cublasLinearForwardFloat32(
        input,
        weights,
        batch_size,
        input_features,
        output_features);

    const kl::Tensor actual = kerneloomLinearForwardCudaFloat32(
        input,
        weights);

    EXPECT_TRUE(kl::test::tensorCompare(
        expected,
        actual,
        float32_absolute_tolerance,
        float32_relative_tolerance));
}

TEST(LinearForwardCudaFloat32, BenchmarkAgainstCuBLAS)
{
    const kl::Tensor input = kl::test::makeRandomTensor(
        kl::Shape{batch_size, input_features},
        kl::DType::Float32,
        kl::Device::cuda(),
        -1.0,
        1.0,
        11);

    const kl::Tensor weights = kl::test::makeRandomTensor(
        kl::Shape{output_features, input_features},
        kl::DType::Float32,
        kl::Device::cuda(),
        -1.0,
        1.0,
        22);

    const double kerneloom_ms = benchmarkGpu(
        kl::Device::cuda(),
        [&]()
        {
            return kerneloomLinearForwardCudaFloat32(
                input,
                weights);
        });

    const double cublas_ms = benchmarkGpu(
        kl::Device::cuda(),
        [&]()
        {
            return kl::test::cublasLinearForwardFloat32(
                input,
                weights,
                batch_size,
                input_features,
                output_features);
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
    const kl::Tensor input = kl::test::makeRandomTensor(
        kl::Shape{batch_size, input_features},
        kl::DType::Float32,
        kl::Device::rocm(),
        -1.0,
        1.0,
        11);

    const kl::Tensor weights = kl::test::makeRandomTensor(
        kl::Shape{output_features, input_features},
        kl::DType::Float32,
        kl::Device::rocm(),
        -1.0,
        1.0,
        22);

    const kl::Tensor expected = kl::test::rocblasLinearForwardFloat32(
        input,
        weights,
        batch_size,
        input_features,
        output_features);

    const kl::Tensor actual = kerneloomLinearForwardRocmFloat32(
        input,
        weights);

    EXPECT_TRUE(kl::test::tensorCompare(
        expected,
        actual,
        float32_absolute_tolerance,
        float32_relative_tolerance));
}

TEST(LinearForwardRocmFloat32, BenchmarkAgainstRocBLAS)
{
    const kl::Tensor input = kl::test::makeRandomTensor(
        kl::Shape{batch_size, input_features},
        kl::DType::Float32,
        kl::Device::rocm(),
        -1.0,
        1.0,
        11);

    const kl::Tensor weights = kl::test::makeRandomTensor(
        kl::Shape{output_features, input_features},
        kl::DType::Float32,
        kl::Device::rocm(),
        -1.0,
        1.0,
        22);

    const double kerneloom_ms = benchmarkGpu(
        kl::Device::rocm(),
        [&]()
        {
            return kerneloomLinearForwardRocmFloat32(
                input,
                weights);
        });

    const double rocblas_ms = benchmarkGpu(
        kl::Device::rocm(),
        [&]()
        {
            return kl::test::rocblasLinearForwardFloat32(
                input,
                weights,
                batch_size,
                input_features,
                output_features);
        });

    kl::test::printBenchmarkComparison(
        "Linear Forward ROCm Float32",
        "Kerneloom ROCm",
        kerneloom_ms,
        "rocBLAS",
        rocblas_ms);
}

#endif // KL_ENABLE_ROCM