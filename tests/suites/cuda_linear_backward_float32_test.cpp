#include <gtest/gtest.h>

#ifdef KL_ENABLE_CUDA

#include "common/benchmark_report.hpp"
#include "common/benchmark_timer.hpp"
#include "common/tensor_compare.hpp"
#include "common/tensor_factory.hpp"
#include "common/test_options.hpp"

#include "options/linear_backward_options.hpp"

#include "vendor/cuda/cublas_linear_backward.cuh"

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>

#include <kernels/cuda/linear/backward_linear_grad_bias_cuda_float32.cuh>
#include <kernels/cuda/linear/backward_linear_grad_input_cuda_float32.cuh>
#include <kernels/cuda/linear/backward_linear_grad_weights_cuda_float32.cuh>

#include <cstddef>
#include <string>

namespace
{

    namespace options =
        kl::test::options::linear_backward;

    void runGradInputCorrectness(
        const options::Shape &shape)
    {
        SCOPED_TRACE(
            std::string("Shape: ") +
            shape.name);

        const auto weights =
            kl::test::makeRandomTensor(
                kl::Shape{
                    shape.output_features,
                    shape.input_features},
                kl::DType::Float32,
                kl::Device::cuda(),
                -1.0,
                1.0,
                22);

        const auto grad_output =
            kl::test::makeRandomTensor(
                kl::Shape{
                    shape.batch_size,
                    shape.output_features},
                kl::DType::Float32,
                kl::Device::cuda(),
                -1.0,
                1.0,
                33);

        kl::Tensor actual_grad_input(
            kl::Shape{
                shape.batch_size,
                shape.input_features},
            kl::DType::Float32,
            kl::Device::cuda());

        kl::backward_linear_grad_input_cuda_float32(
            weights,
            grad_output,
            actual_grad_input);

        const auto expected_grad_input =
            kl::test::cublasLinearBackwardGradInputFloat32(
                weights,
                grad_output);

        EXPECT_TRUE(kl::test::tensorCompare(
            expected_grad_input,
            actual_grad_input,
            options::float32_absolute_tolerance,
            options::float32_relative_tolerance,
            kl::test::options::tensor_compare::max_mismatch_ratio));
    }

    void runGradWeightsCorrectness(
        const options::Shape &shape)
    {
        SCOPED_TRACE(
            std::string("Shape: ") +
            shape.name);

        const auto input =
            kl::test::makeRandomTensor(
                kl::Shape{
                    shape.batch_size,
                    shape.input_features},
                kl::DType::Float32,
                kl::Device::cuda(),
                -1.0,
                1.0,
                11);

        const auto grad_output =
            kl::test::makeRandomTensor(
                kl::Shape{
                    shape.batch_size,
                    shape.output_features},
                kl::DType::Float32,
                kl::Device::cuda(),
                -1.0,
                1.0,
                33);

        kl::Tensor actual_grad_weights(
            kl::Shape{
                shape.output_features,
                shape.input_features},
            kl::DType::Float32,
            kl::Device::cuda());

        kl::backward_linear_grad_weights_cuda_float32(
            input,
            grad_output,
            actual_grad_weights);

        const auto expected_grad_weights =
            kl::test::cublasLinearBackwardGradWeightsFloat32(
                input,
                grad_output);

        EXPECT_TRUE(kl::test::tensorCompare(
            expected_grad_weights,
            actual_grad_weights,
            options::float32_absolute_tolerance,
            options::float32_relative_tolerance,
            kl::test::options::tensor_compare::max_mismatch_ratio));
    }

    void runGradBiasCorrectness(
        const options::BiasShape &shape)
    {
        SCOPED_TRACE(
            std::string("Shape: ") +
            shape.name);

        const auto grad_output =
            kl::test::makeRandomTensor(
                kl::Shape{
                    shape.batch_size,
                    shape.output_features},
                kl::DType::Float32,
                kl::Device::cuda(),
                -1.0,
                1.0,
                33);

        kl::Tensor actual_grad_bias(
            kl::Shape{
                shape.output_features},
            kl::DType::Float32,
            kl::Device::cuda());

        kl::backward_linear_grad_bias_cuda_float32(
            grad_output,
            actual_grad_bias);

        const auto expected_grad_bias =
            kl::test::cublasLinearBackwardGradBiasFloat32(
                grad_output);

        EXPECT_TRUE(kl::test::tensorCompare(
            expected_grad_bias,
            actual_grad_bias,
            options::float32_absolute_tolerance,
            options::float32_relative_tolerance,
            kl::test::options::tensor_compare::max_mismatch_ratio));
    }

    void runGradInputBenchmark(
        const options::Shape &shape)
    {
        const auto weights =
            kl::test::makeRandomTensor(
                kl::Shape{
                    shape.output_features,
                    shape.input_features},
                kl::DType::Float32,
                kl::Device::cuda(),
                -1.0,
                1.0,
                22);

        const auto grad_output =
            kl::test::makeRandomTensor(
                kl::Shape{
                    shape.batch_size,
                    shape.output_features},
                kl::DType::Float32,
                kl::Device::cuda(),
                -1.0,
                1.0,
                33);

        kl::Tensor kerneloom_grad_input(
            kl::Shape{
                shape.batch_size,
                shape.input_features},
            kl::DType::Float32,
            kl::Device::cuda());

        kl::Tensor cublas_grad_input(
            kl::Shape{
                shape.batch_size,
                shape.input_features},
            kl::DType::Float32,
            kl::Device::cuda());

        kl::test::CublasHandle cublas_handle;

        const double kerneloom_ms =
            kl::test::benchmarkGpu(
                kl::Device::cuda(),
                options::warmup_iterations,
                options::measured_iterations,
                kl::test::options::benchmark::print_each_iteration,
                "Kerneloom CUDA",
                [&]()
                {
                    kl::backward_linear_grad_input_cuda_float32(
                        weights,
                        grad_output,
                        kerneloom_grad_input);
                });

        const double cublas_ms =
            kl::test::benchmarkGpu(
                kl::Device::cuda(),
                options::warmup_iterations,
                options::measured_iterations,
                kl::test::options::benchmark::print_each_iteration,
                "cuBLAS",
                [&]()
                {
                    kl::test::cublasLinearBackwardGradInputFloat32(
                        cublas_handle,
                        weights,
                        grad_output,
                        cublas_grad_input);
                });

        EXPECT_TRUE(kl::test::tensorCompare(
            cublas_grad_input,
            kerneloom_grad_input,
            options::float32_absolute_tolerance,
            options::float32_relative_tolerance,
            kl::test::options::tensor_compare::max_mismatch_ratio));

        const std::string benchmark_name =
            std::string("Linear Backward Grad Input CUDA Float32 ") +
            shape.name;

        kl::test::printBenchmarkComparison(
            benchmark_name.c_str(),
            "Kerneloom CUDA",
            kerneloom_ms,
            "cuBLAS",
            cublas_ms);
    }

    void runGradWeightsBenchmark(
        const options::Shape &shape)
    {
        const auto input =
            kl::test::makeRandomTensor(
                kl::Shape{
                    shape.batch_size,
                    shape.input_features},
                kl::DType::Float32,
                kl::Device::cuda(),
                -1.0,
                1.0,
                11);

        const auto grad_output =
            kl::test::makeRandomTensor(
                kl::Shape{
                    shape.batch_size,
                    shape.output_features},
                kl::DType::Float32,
                kl::Device::cuda(),
                -1.0,
                1.0,
                33);

        kl::Tensor kerneloom_grad_weights(
            kl::Shape{
                shape.output_features,
                shape.input_features},
            kl::DType::Float32,
            kl::Device::cuda());

        kl::Tensor cublas_grad_weights(
            kl::Shape{
                shape.output_features,
                shape.input_features},
            kl::DType::Float32,
            kl::Device::cuda());

        kl::test::CublasHandle cublas_handle;

        const double kerneloom_ms =
            kl::test::benchmarkGpu(
                kl::Device::cuda(),
                options::warmup_iterations,
                options::measured_iterations,
                kl::test::options::benchmark::print_each_iteration,
                "Kerneloom CUDA",
                [&]()
                {
                    kl::backward_linear_grad_weights_cuda_float32(
                        input,
                        grad_output,
                        kerneloom_grad_weights);
                });

        const double cublas_ms =
            kl::test::benchmarkGpu(
                kl::Device::cuda(),
                options::warmup_iterations,
                options::measured_iterations,
                kl::test::options::benchmark::print_each_iteration,
                "cuBLAS",
                [&]()
                {
                    kl::test::cublasLinearBackwardGradWeightsFloat32(
                        cublas_handle,
                        input,
                        grad_output,
                        cublas_grad_weights);
                });

        EXPECT_TRUE(kl::test::tensorCompare(
            cublas_grad_weights,
            kerneloom_grad_weights,
            options::float32_absolute_tolerance,
            options::float32_relative_tolerance,
            kl::test::options::tensor_compare::max_mismatch_ratio));

        const std::string benchmark_name =
            std::string("Linear Backward Grad Weights CUDA Float32 ") +
            shape.name;

        kl::test::printBenchmarkComparison(
            benchmark_name.c_str(),
            "Kerneloom CUDA",
            kerneloom_ms,
            "cuBLAS",
            cublas_ms);
    }

    void runGradBiasBenchmark(
        const options::BiasShape &shape)
    {
        const auto grad_output =
            kl::test::makeRandomTensor(
                kl::Shape{
                    shape.batch_size,
                    shape.output_features},
                kl::DType::Float32,
                kl::Device::cuda(),
                -1.0,
                1.0,
                33);

        kl::Tensor kerneloom_grad_bias(
            kl::Shape{
                shape.output_features},
            kl::DType::Float32,
            kl::Device::cuda());

        kl::Tensor cublas_grad_bias(
            kl::Shape{
                shape.output_features},
            kl::DType::Float32,
            kl::Device::cuda());

        const auto ones =
            kl::test::makeOnesCudaFloat32(
                shape.batch_size);

        kl::test::CublasHandle cublas_handle;

        const double kerneloom_ms =
            kl::test::benchmarkGpu(
                kl::Device::cuda(),
                options::warmup_iterations,
                options::measured_iterations,
                kl::test::options::benchmark::print_each_iteration,
                "Kerneloom CUDA",
                [&]()
                {
                    kl::backward_linear_grad_bias_cuda_float32(
                        grad_output,
                        kerneloom_grad_bias);
                });

        const double cublas_ms =
            kl::test::benchmarkGpu(
                kl::Device::cuda(),
                options::warmup_iterations,
                options::measured_iterations,
                kl::test::options::benchmark::print_each_iteration,
                "cuBLAS",
                [&]()
                {
                    kl::test::cublasLinearBackwardGradBiasFloat32(
                        cublas_handle,
                        grad_output,
                        ones,
                        cublas_grad_bias);
                });

        EXPECT_TRUE(kl::test::tensorCompare(
            cublas_grad_bias,
            kerneloom_grad_bias,
            options::float32_absolute_tolerance,
            options::float32_relative_tolerance,
            kl::test::options::tensor_compare::max_mismatch_ratio));

        const std::string benchmark_name =
            std::string("Linear Backward Grad Bias CUDA Float32 ") +
            shape.name;

        kl::test::printBenchmarkComparison(
            benchmark_name.c_str(),
            "Kerneloom CUDA",
            kerneloom_ms,
            "cuBLAS",
            cublas_ms);
    }

}

TEST(LinearBackwardCudaFloat32, GradInputMatchesCuBLAS)
{
    for (const auto &shape : options::correctness_shapes)
    {
        runGradInputCorrectness(
            shape);
    }
}

TEST(LinearBackwardCudaFloat32, GradWeightsMatchesCuBLAS)
{
    for (const auto &shape : options::correctness_shapes)
    {
        runGradWeightsCorrectness(
            shape);
    }
}

TEST(LinearBackwardCudaFloat32, GradBiasMatchesCuBLAS)
{
    for (const auto &shape : options::grad_bias_correctness_shapes)
    {
        runGradBiasCorrectness(
            shape);
    }
}

TEST(LinearBackwardCudaFloat32, GradInputBenchmarkAgainstCuBLAS)
{
    for (const auto &shape : options::benchmark_shapes)
    {
        runGradInputBenchmark(
            shape);
    }
}

TEST(LinearBackwardCudaFloat32, GradWeightsBenchmarkAgainstCuBLAS)
{
    for (const auto &shape : options::benchmark_shapes)
    {
        runGradWeightsBenchmark(
            shape);
    }
}

TEST(LinearBackwardCudaFloat32, GradBiasBenchmarkAgainstCuBLAS)
{
    for (const auto &shape : options::grad_bias_benchmark_shapes)
    {
        runGradBiasBenchmark(
            shape);
    }
}

#endif // KL_ENABLE_CUDA