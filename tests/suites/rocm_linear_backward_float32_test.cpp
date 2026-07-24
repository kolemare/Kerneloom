#include <gtest/gtest.h>

#ifdef KL_ENABLE_ROCM

#include "common/benchmark_report.hpp"
#include "common/benchmark_timer.hpp"
#include "common/tensor_compare.hpp"
#include "common/tensor_factory.hpp"
#include "common/test_options.hpp"

#include "vendor/rocm/rocblas_linear_backward.hiph"

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>

#include <kernels/rocm/linear/backward_linear_grad_bias_rocm_float32.hiph>
#include <kernels/rocm/linear/backward_linear_grad_input_rocm_float32.hiph>
#include <kernels/rocm/linear/backward_linear_grad_weights_rocm_float32.hiph>

#include <cstddef>

namespace
{

    namespace options =
        kl::test::options::linear_backward_float32;

    void runGradInputCorrectness(
        std::size_t batch_size,
        std::size_t input_features,
        std::size_t output_features)
    {
        const auto weights =
            kl::test::makeRandomTensor(
                kl::Shape{
                    output_features,
                    input_features},
                kl::DType::Float32,
                kl::Device::rocm(),
                -1.0,
                1.0,
                22);

        const auto grad_output =
            kl::test::makeRandomTensor(
                kl::Shape{
                    batch_size,
                    output_features},
                kl::DType::Float32,
                kl::Device::rocm(),
                -1.0,
                1.0,
                33);

        kl::Tensor actual_grad_input(
            kl::Shape{
                batch_size,
                input_features},
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
            options::absolute_tolerance,
            options::relative_tolerance));
    }

    void runGradWeightsCorrectness(
        std::size_t batch_size,
        std::size_t input_features,
        std::size_t output_features)
    {
        const auto input =
            kl::test::makeRandomTensor(
                kl::Shape{
                    batch_size,
                    input_features},
                kl::DType::Float32,
                kl::Device::rocm(),
                -1.0,
                1.0,
                11);

        const auto grad_output =
            kl::test::makeRandomTensor(
                kl::Shape{
                    batch_size,
                    output_features},
                kl::DType::Float32,
                kl::Device::rocm(),
                -1.0,
                1.0,
                33);

        kl::Tensor actual_grad_weights(
            kl::Shape{
                output_features,
                input_features},
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
            options::absolute_tolerance,
            options::relative_tolerance));
    }

    void runGradBiasCorrectness(
        std::size_t batch_size,
        std::size_t output_features)
    {
        const auto grad_output =
            kl::test::makeRandomTensor(
                kl::Shape{
                    batch_size,
                    output_features},
                kl::DType::Float32,
                kl::Device::rocm(),
                -1.0,
                1.0,
                33);

        kl::Tensor actual_grad_bias(
            kl::Shape{
                output_features},
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
            options::absolute_tolerance,
            options::relative_tolerance));
    }

    void runGradInputBenchmark(
        const char *benchmark_name,
        std::size_t batch_size,
        std::size_t input_features,
        std::size_t output_features)
    {
        const auto weights =
            kl::test::makeRandomTensor(
                kl::Shape{
                    output_features,
                    input_features},
                kl::DType::Float32,
                kl::Device::rocm(),
                -1.0,
                1.0,
                22);

        const auto grad_output =
            kl::test::makeRandomTensor(
                kl::Shape{
                    batch_size,
                    output_features},
                kl::DType::Float32,
                kl::Device::rocm(),
                -1.0,
                1.0,
                33);

        kl::Tensor kerneloom_grad_input(
            kl::Shape{
                batch_size,
                input_features},
            kl::DType::Float32,
            kl::Device::rocm());

        kl::Tensor rocblas_grad_input(
            kl::Shape{
                batch_size,
                input_features},
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
            options::absolute_tolerance,
            options::relative_tolerance));

        kl::test::printBenchmarkComparison(
            benchmark_name,
            "Kerneloom ROCm",
            kerneloom_ms,
            "rocBLAS",
            rocblas_ms);
    }

    void runGradWeightsBenchmark(
        const char *benchmark_name,
        std::size_t batch_size,
        std::size_t input_features,
        std::size_t output_features)
    {
        const auto input =
            kl::test::makeRandomTensor(
                kl::Shape{
                    batch_size,
                    input_features},
                kl::DType::Float32,
                kl::Device::rocm(),
                -1.0,
                1.0,
                11);

        const auto grad_output =
            kl::test::makeRandomTensor(
                kl::Shape{
                    batch_size,
                    output_features},
                kl::DType::Float32,
                kl::Device::rocm(),
                -1.0,
                1.0,
                33);

        kl::Tensor kerneloom_grad_weights(
            kl::Shape{
                output_features,
                input_features},
            kl::DType::Float32,
            kl::Device::rocm());

        kl::Tensor rocblas_grad_weights(
            kl::Shape{
                output_features,
                input_features},
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
            options::absolute_tolerance,
            options::relative_tolerance));

        kl::test::printBenchmarkComparison(
            benchmark_name,
            "Kerneloom ROCm",
            kerneloom_ms,
            "rocBLAS",
            rocblas_ms);
    }

    void runGradBiasBenchmark(
        const char *benchmark_name,
        std::size_t batch_size,
        std::size_t output_features)
    {
        const auto grad_output =
            kl::test::makeRandomTensor(
                kl::Shape{
                    batch_size,
                    output_features},
                kl::DType::Float32,
                kl::Device::rocm(),
                -1.0,
                1.0,
                33);

        kl::Tensor kerneloom_grad_bias(
            kl::Shape{
                output_features},
            kl::DType::Float32,
            kl::Device::rocm());

        kl::Tensor rocblas_grad_bias(
            kl::Shape{
                output_features},
            kl::DType::Float32,
            kl::Device::rocm());

        const auto ones =
            kl::test::makeOnesRocmFloat32(
                batch_size);

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
            options::absolute_tolerance,
            options::relative_tolerance));

        kl::test::printBenchmarkComparison(
            benchmark_name,
            "Kerneloom ROCm",
            kerneloom_ms,
            "rocBLAS",
            rocblas_ms);
    }

}

TEST(LinearBackwardRocmFloat32, GradInputMatchesRocBLAS_LargeNonSquare)
{
    runGradInputCorrectness(
        options::large_non_square::batch_size,
        options::large_non_square::input_features,
        options::large_non_square::output_features);
}

TEST(LinearBackwardRocmFloat32, GradWeightsMatchesRocBLAS_LargeNonSquare)
{
    runGradWeightsCorrectness(
        options::large_non_square::batch_size,
        options::large_non_square::input_features,
        options::large_non_square::output_features);
}

TEST(LinearBackwardRocmFloat32, GradBiasMatchesRocBLAS_LargeNonSquare)
{
    runGradBiasCorrectness(
        options::large_non_square::batch_size,
        options::large_non_square::output_features);
}

TEST(LinearBackwardRocmFloat32, GradInputBenchmarkAgainstRocBLAS_LargeNonSquare)
{
    runGradInputBenchmark(
        "Linear Backward Grad Input ROCm Float32 Large Non-Square",
        options::large_non_square::batch_size,
        options::large_non_square::input_features,
        options::large_non_square::output_features);
}

TEST(LinearBackwardRocmFloat32, GradWeightsBenchmarkAgainstRocBLAS_LargeNonSquare)
{
    runGradWeightsBenchmark(
        "Linear Backward Grad Weights ROCm Float32 Large Non-Square",
        options::large_non_square::batch_size,
        options::large_non_square::input_features,
        options::large_non_square::output_features);
}

TEST(LinearBackwardRocmFloat32, GradBiasBenchmarkAgainstRocBLAS_LargeNonSquare)
{
    runGradBiasBenchmark(
        "Linear Backward Grad Bias ROCm Float32 Large Non-Square",
        options::large_non_square::batch_size,
        options::large_non_square::output_features);
}

TEST(LinearBackwardRocmFloat32, GradInputMatchesRocBLAS_DemandingOddShape)
{
    runGradInputCorrectness(
        options::demanding_odd_shape::batch_size,
        options::demanding_odd_shape::input_features,
        options::demanding_odd_shape::output_features);
}

TEST(LinearBackwardRocmFloat32, GradWeightsMatchesRocBLAS_DemandingOddShape)
{
    runGradWeightsCorrectness(
        options::demanding_odd_shape::batch_size,
        options::demanding_odd_shape::input_features,
        options::demanding_odd_shape::output_features);
}

TEST(LinearBackwardRocmFloat32, GradBiasMatchesRocBLAS_DemandingOddShape)
{
    runGradBiasCorrectness(
        options::demanding_odd_shape::batch_size,
        options::demanding_odd_shape::output_features);
}

TEST(LinearBackwardRocmFloat32, GradInputBenchmarkAgainstRocBLAS_DemandingOddShape)
{
    runGradInputBenchmark(
        "Linear Backward Grad Input ROCm Float32 Demanding Odd Shape",
        options::demanding_odd_shape::batch_size,
        options::demanding_odd_shape::input_features,
        options::demanding_odd_shape::output_features);
}

TEST(LinearBackwardRocmFloat32, GradWeightsBenchmarkAgainstRocBLAS_DemandingOddShape)
{
    runGradWeightsBenchmark(
        "Linear Backward Grad Weights ROCm Float32 Demanding Odd Shape",
        options::demanding_odd_shape::batch_size,
        options::demanding_odd_shape::input_features,
        options::demanding_odd_shape::output_features);
}

TEST(LinearBackwardRocmFloat32, GradBiasBenchmarkAgainstRocBLAS_DemandingOddShape)
{
    runGradBiasBenchmark(
        "Linear Backward Grad Bias ROCm Float32 Demanding Odd Shape",
        options::demanding_odd_shape::batch_size,
        options::demanding_odd_shape::output_features);
}

#endif // KL_ENABLE_ROCM