#include <gtest/gtest.h>

#ifdef KL_ENABLE_ROCM

#include "common/benchmark_report.hpp"
#include "common/benchmark_timer.hpp"
#include "common/tensor_compare.hpp"
#include "common/tensor_factory.hpp"
#include "common/test_options.hpp"

#include "options/linear_backward_options.hpp"

#include "vendor/rocm/rocblas_linear_backward.hiph"

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>

#include <kernels/rocm/linear/backward_linear_grad_bias_rocm_float32.hiph>
#include <kernels/rocm/linear/backward_linear_grad_input_rocm_float32.hiph>
#include <kernels/rocm/linear/backward_linear_grad_weights_rocm_float32.hiph>

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
                kl::Device::rocm(),
                -1.0,
                1.0,
                22);

        const auto grad_output =
            kl::test::makeRandomTensor(
                kl::Shape{
                    shape.batch_size,
                    shape.output_features},
                kl::DType::Float32,
                kl::Device::rocm(),
                -1.0,
                1.0,
                33);

        kl::Tensor actual_grad_input(
            kl::Shape{
                shape.batch_size,
                shape.input_features},
            kl::DType::Float32,
            kl::Device::rocm());

        kl::backward_linear_grad_input_rocm_float32(
            weights,
            grad_output,
            actual_grad_input);

        const auto expected_grad_input =
            kl::test::rocblasLinearBackwardGradInputFloat32(
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
                kl::Device::rocm(),
                -1.0,
                1.0,
                11);

        const auto grad_output =
            kl::test::makeRandomTensor(
                kl::Shape{
                    shape.batch_size,
                    shape.output_features},
                kl::DType::Float32,
                kl::Device::rocm(),
                -1.0,
                1.0,
                33);

        kl::Tensor actual_grad_weights(
            kl::Shape{
                shape.output_features,
                shape.input_features},
            kl::DType::Float32,
            kl::Device::rocm());

        kl::backward_linear_grad_weights_rocm_float32(
            input,
            grad_output,
            actual_grad_weights);

        const auto expected_grad_weights =
            kl::test::rocblasLinearBackwardGradWeightsFloat32(
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
                kl::Device::rocm(),
                -1.0,
                1.0,
                33);

        kl::Tensor actual_grad_bias(
            kl::Shape{
                shape.output_features},
            kl::DType::Float32,
            kl::Device::rocm());

        kl::backward_linear_grad_bias_rocm_float32(
            grad_output,
            actual_grad_bias);

        const auto expected_grad_bias =
            kl::test::rocblasLinearBackwardGradBiasFloat32(
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
                kl::Device::rocm(),
                -1.0,
                1.0,
                22);

        const auto grad_output =
            kl::test::makeRandomTensor(
                kl::Shape{
                    shape.batch_size,
                    shape.output_features},
                kl::DType::Float32,
                kl::Device::rocm(),
                -1.0,
                1.0,
                33);

        kl::Tensor kerneloom_grad_input(
            kl::Shape{
                shape.batch_size,
                shape.input_features},
            kl::DType::Float32,
            kl::Device::rocm());

        kl::Tensor rocblas_grad_input(
            kl::Shape{
                shape.batch_size,
                shape.input_features},
            kl::DType::Float32,
            kl::Device::rocm());

        kl::test::RocblasHandle rocblas_handle;

        const double kerneloom_ms =
            kl::test::benchmarkGpu(
                kl::Device::rocm(),
                options::warmup_iterations,
                options::measured_iterations,
                kl::test::options::benchmark::print_each_iteration,
                "Kerneloom ROCm",
                [&]()
                {
                    kl::backward_linear_grad_input_rocm_float32(
                        weights,
                        grad_output,
                        kerneloom_grad_input);
                });

        const double rocblas_ms =
            kl::test::benchmarkGpu(
                kl::Device::rocm(),
                options::warmup_iterations,
                options::measured_iterations,
                kl::test::options::benchmark::print_each_iteration,
                "rocBLAS",
                [&]()
                {
                    kl::test::rocblasLinearBackwardGradInputFloat32(
                        rocblas_handle,
                        weights,
                        grad_output,
                        rocblas_grad_input);
                });

        EXPECT_TRUE(kl::test::tensorCompare(
            rocblas_grad_input,
            kerneloom_grad_input,
            options::float32_absolute_tolerance,
            options::float32_relative_tolerance,
            kl::test::options::tensor_compare::max_mismatch_ratio));

        const std::string benchmark_name =
            std::string("Linear Backward Grad Input ROCm Float32 ") +
            shape.name;

        kl::test::printBenchmarkComparison(
            benchmark_name.c_str(),
            "Kerneloom ROCm",
            kerneloom_ms,
            "rocBLAS",
            rocblas_ms);
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
                kl::Device::rocm(),
                -1.0,
                1.0,
                11);

        const auto grad_output =
            kl::test::makeRandomTensor(
                kl::Shape{
                    shape.batch_size,
                    shape.output_features},
                kl::DType::Float32,
                kl::Device::rocm(),
                -1.0,
                1.0,
                33);

        kl::Tensor kerneloom_grad_weights(
            kl::Shape{
                shape.output_features,
                shape.input_features},
            kl::DType::Float32,
            kl::Device::rocm());

        kl::Tensor rocblas_grad_weights(
            kl::Shape{
                shape.output_features,
                shape.input_features},
            kl::DType::Float32,
            kl::Device::rocm());

        kl::test::RocblasHandle rocblas_handle;

        const double kerneloom_ms =
            kl::test::benchmarkGpu(
                kl::Device::rocm(),
                options::warmup_iterations,
                options::measured_iterations,
                kl::test::options::benchmark::print_each_iteration,
                "Kerneloom ROCm",
                [&]()
                {
                    kl::backward_linear_grad_weights_rocm_float32(
                        input,
                        grad_output,
                        kerneloom_grad_weights);
                });

        const double rocblas_ms =
            kl::test::benchmarkGpu(
                kl::Device::rocm(),
                options::warmup_iterations,
                options::measured_iterations,
                kl::test::options::benchmark::print_each_iteration,
                "rocBLAS",
                [&]()
                {
                    kl::test::rocblasLinearBackwardGradWeightsFloat32(
                        rocblas_handle,
                        input,
                        grad_output,
                        rocblas_grad_weights);
                });

        EXPECT_TRUE(kl::test::tensorCompare(
            rocblas_grad_weights,
            kerneloom_grad_weights,
            options::float32_absolute_tolerance,
            options::float32_relative_tolerance,
            kl::test::options::tensor_compare::max_mismatch_ratio));

        const std::string benchmark_name =
            std::string("Linear Backward Grad Weights ROCm Float32 ") +
            shape.name;

        kl::test::printBenchmarkComparison(
            benchmark_name.c_str(),
            "Kerneloom ROCm",
            kerneloom_ms,
            "rocBLAS",
            rocblas_ms);
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
                kl::Device::rocm(),
                -1.0,
                1.0,
                33);

        kl::Tensor kerneloom_grad_bias(
            kl::Shape{
                shape.output_features},
            kl::DType::Float32,
            kl::Device::rocm());

        kl::Tensor rocblas_grad_bias(
            kl::Shape{
                shape.output_features},
            kl::DType::Float32,
            kl::Device::rocm());

        const auto ones =
            kl::test::makeOnesRocmFloat32(
                shape.batch_size);

        kl::test::RocblasHandle rocblas_handle;

        const double kerneloom_ms =
            kl::test::benchmarkGpu(
                kl::Device::rocm(),
                options::warmup_iterations,
                options::measured_iterations,
                kl::test::options::benchmark::print_each_iteration,
                "Kerneloom ROCm",
                [&]()
                {
                    kl::backward_linear_grad_bias_rocm_float32(
                        grad_output,
                        kerneloom_grad_bias);
                });

        const double rocblas_ms =
            kl::test::benchmarkGpu(
                kl::Device::rocm(),
                options::warmup_iterations,
                options::measured_iterations,
                kl::test::options::benchmark::print_each_iteration,
                "rocBLAS",
                [&]()
                {
                    kl::test::rocblasLinearBackwardGradBiasFloat32(
                        rocblas_handle,
                        grad_output,
                        ones,
                        rocblas_grad_bias);
                });

        EXPECT_TRUE(kl::test::tensorCompare(
            rocblas_grad_bias,
            kerneloom_grad_bias,
            options::float32_absolute_tolerance,
            options::float32_relative_tolerance,
            kl::test::options::tensor_compare::max_mismatch_ratio));

        const std::string benchmark_name =
            std::string("Linear Backward Grad Bias ROCm Float32 ") +
            shape.name;

        kl::test::printBenchmarkComparison(
            benchmark_name.c_str(),
            "Kerneloom ROCm",
            kerneloom_ms,
            "rocBLAS",
            rocblas_ms);
    }

}

TEST(LinearBackwardRocmFloat32, GradInputMatchesRocBLAS)
{
    for (const auto &shape : options::correctness_shapes)
    {
        runGradInputCorrectness(
            shape);
    }
}

TEST(LinearBackwardRocmFloat32, GradWeightsMatchesRocBLAS)
{
    for (const auto &shape : options::correctness_shapes)
    {
        runGradWeightsCorrectness(
            shape);
    }
}

TEST(LinearBackwardRocmFloat32, GradBiasMatchesRocBLAS)
{
    for (const auto &shape : options::grad_bias_correctness_shapes)
    {
        runGradBiasCorrectness(
            shape);
    }
}

TEST(LinearBackwardRocmFloat32, GradInputBenchmarkAgainstRocBLAS)
{
    for (const auto &shape : options::benchmark_shapes)
    {
        runGradInputBenchmark(
            shape);
    }
}

TEST(LinearBackwardRocmFloat32, GradWeightsBenchmarkAgainstRocBLAS)
{
    for (const auto &shape : options::benchmark_shapes)
    {
        runGradWeightsBenchmark(
            shape);
    }
}

TEST(LinearBackwardRocmFloat32, GradBiasBenchmarkAgainstRocBLAS)
{
    for (const auto &shape : options::grad_bias_benchmark_shapes)
    {
        runGradBiasBenchmark(
            shape);
    }
}

#endif // KL_ENABLE_ROCM