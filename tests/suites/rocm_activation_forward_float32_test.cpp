#include <gtest/gtest.h>

#ifdef KL_ENABLE_ROCM

#include "common/benchmark_report.hpp"
#include "common/benchmark_timer.hpp"
#include "common/tensor_compare.hpp"
#include "common/tensor_factory.hpp"
#include "common/test_options.hpp"

#include "options/activation_forward_options.hpp"

#include "vendor/rocm/miopen_activation.hiph"

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>

#include <kernels/rocm/activation/relu_rocm_float32.hiph>
#include <kernels/rocm/activation/sigmoid_rocm_float32.hiph>
#include <kernels/rocm/activation/softmax_rocm_float32.hiph>
#include <kernels/rocm/activation/tanh_rocm_float32.hiph>

#include <miopen/miopen.h>

#include <cstddef>
#include <string>

namespace
{

    namespace options =
        kl::test::options::activation_forward;

    using ActivationKernel =
        void (*)(kl::Tensor &);

    void runElementwiseCorrectness(
        const options::ElementwiseShape &shape,
        double lower,
        double upper,
        int seed,
        miopenActivationMode_t miopen_mode,
        ActivationKernel kernel,
        double absolute_tolerance,
        double relative_tolerance)
    {
        SCOPED_TRACE(
            std::string("Shape: ") +
            shape.name);

        auto actual =
            kl::test::makeRandomTensor(
                kl::Shape{
                    1,
                    1,
                    1,
                    shape.element_count},
                kl::DType::Float32,
                kl::Device::rocm(),
                lower,
                upper,
                seed);

        auto expected =
            kl::test::makeRandomTensor(
                kl::Shape{
                    1,
                    1,
                    1,
                    shape.element_count},
                kl::DType::Float32,
                kl::Device::rocm(),
                lower,
                upper,
                seed);

        kernel(
            actual);

        expected =
            kl::test::miopenActivationForwardFloat32(
                expected,
                miopen_mode);

        EXPECT_TRUE(kl::test::tensorCompare(
            expected,
            actual,
            absolute_tolerance,
            relative_tolerance));
    }

    void runSoftmaxCorrectness(
        const options::SoftmaxShape &shape)
    {
        SCOPED_TRACE(
            std::string("Shape: ") +
            shape.name);

        auto actual =
            kl::test::makeRandomTensor(
                kl::Shape{
                    shape.batch_size,
                    shape.class_count,
                    1,
                    1},
                kl::DType::Float32,
                kl::Device::rocm(),
                -10.0,
                10.0,
                44);

        auto expected =
            kl::test::makeRandomTensor(
                kl::Shape{
                    shape.batch_size,
                    shape.class_count,
                    1,
                    1},
                kl::DType::Float32,
                kl::Device::rocm(),
                -10.0,
                10.0,
                44);

        kl::softmax_rocm_float32(
            actual);

        expected =
            kl::test::miopenSoftmaxForwardFloat32(
                expected);

        EXPECT_TRUE(kl::test::tensorCompare(
            expected,
            actual,
            options::softmax_float32_absolute_tolerance,
            options::softmax_float32_relative_tolerance));
    }

    void runElementwiseBenchmark(
        const char *benchmark_name,
        const options::ElementwiseShape &shape,
        double lower,
        double upper,
        int seed,
        miopenActivationMode_t miopen_mode,
        ActivationKernel kernel)
    {
        auto kerneloom_tensor =
            kl::test::makeRandomTensor(
                kl::Shape{
                    1,
                    1,
                    1,
                    shape.element_count},
                kl::DType::Float32,
                kl::Device::rocm(),
                lower,
                upper,
                seed);

        auto miopen_tensor =
            kl::test::makeRandomTensor(
                kl::Shape{
                    1,
                    1,
                    1,
                    shape.element_count},
                kl::DType::Float32,
                kl::Device::rocm(),
                lower,
                upper,
                seed);

        kl::test::MiopenHandle miopen_handle;

        kl::test::MiopenTensorDescriptor miopen_tensor_descriptor(
            miopen_tensor);

        kl::test::MiopenActivationDescriptor miopen_activation_descriptor(
            miopen_mode);

        const double kerneloom_ms =
            kl::test::benchmarkGpu(
                kl::Device::rocm(),
                options::warmup_iterations,
                options::measured_iterations,
                kl::test::options::benchmark::print_each_iteration,
                "Kerneloom ROCm",
                [&]()
                {
                    kernel(
                        kerneloom_tensor);
                });

        const double miopen_ms =
            kl::test::benchmarkGpu(
                kl::Device::rocm(),
                options::warmup_iterations,
                options::measured_iterations,
                kl::test::options::benchmark::print_each_iteration,
                "MIOpen",
                [&]()
                {
                    kl::test::miopenActivationForwardFloat32(
                        miopen_handle,
                        miopen_tensor_descriptor,
                        miopen_activation_descriptor,
                        miopen_tensor,
                        miopen_tensor);
                });

        EXPECT_TRUE(kl::test::tensorCompare(
            miopen_tensor,
            kerneloom_tensor,
            options::elementwise_float32_absolute_tolerance,
            options::elementwise_float32_relative_tolerance));

        const std::string full_benchmark_name =
            std::string(benchmark_name) +
            " " +
            shape.name;

        kl::test::printBenchmarkComparison(
            full_benchmark_name.c_str(),
            "Kerneloom ROCm",
            kerneloom_ms,
            "MIOpen",
            miopen_ms);
    }

    void runSoftmaxBenchmark(
        const char *benchmark_name,
        const options::SoftmaxShape &shape)
    {
        auto kerneloom_tensor =
            kl::test::makeRandomTensor(
                kl::Shape{
                    shape.batch_size,
                    shape.class_count,
                    1,
                    1},
                kl::DType::Float32,
                kl::Device::rocm(),
                -10.0,
                10.0,
                44);

        auto miopen_tensor =
            kl::test::makeRandomTensor(
                kl::Shape{
                    shape.batch_size,
                    shape.class_count,
                    1,
                    1},
                kl::DType::Float32,
                kl::Device::rocm(),
                -10.0,
                10.0,
                44);

        kl::test::MiopenHandle miopen_handle;

        kl::test::MiopenTensorDescriptor miopen_tensor_descriptor(
            miopen_tensor);

        const double kerneloom_ms =
            kl::test::benchmarkGpu(
                kl::Device::rocm(),
                options::warmup_iterations,
                options::measured_iterations,
                kl::test::options::benchmark::print_each_iteration,
                "Kerneloom ROCm",
                [&]()
                {
                    kl::softmax_rocm_float32(
                        kerneloom_tensor);
                });

        const double miopen_ms =
            kl::test::benchmarkGpu(
                kl::Device::rocm(),
                options::warmup_iterations,
                options::measured_iterations,
                kl::test::options::benchmark::print_each_iteration,
                "MIOpen",
                [&]()
                {
                    kl::test::miopenSoftmaxForwardFloat32(
                        miopen_handle,
                        miopen_tensor_descriptor,
                        miopen_tensor,
                        miopen_tensor);
                });

        EXPECT_TRUE(kl::test::tensorCompare(
            miopen_tensor,
            kerneloom_tensor,
            options::softmax_float32_absolute_tolerance,
            options::softmax_float32_relative_tolerance));

        const std::string full_benchmark_name =
            std::string(benchmark_name) +
            " " +
            shape.name;

        kl::test::printBenchmarkComparison(
            full_benchmark_name.c_str(),
            "Kerneloom ROCm",
            kerneloom_ms,
            "MIOpen",
            miopen_ms);
    }

}

TEST(ActivationForwardRocmFloat32, ReluCorrectness)
{
    for (const auto &shape : options::elementwise_correctness_shapes)
    {
        runElementwiseCorrectness(
            shape,
            -10.0,
            10.0,
            11,
            miopenActivationRELU,
            kl::relu_rocm_float32,
            options::elementwise_float32_absolute_tolerance,
            options::elementwise_float32_relative_tolerance);
    }
}

TEST(ActivationForwardRocmFloat32, SigmoidCorrectness)
{
    for (const auto &shape : options::elementwise_correctness_shapes)
    {
        runElementwiseCorrectness(
            shape,
            -8.0,
            8.0,
            22,
            miopenActivationLOGISTIC,
            kl::sigmoid_rocm_float32,
            options::elementwise_float32_absolute_tolerance,
            options::elementwise_float32_relative_tolerance);
    }
}

TEST(ActivationForwardRocmFloat32, TanhCorrectness)
{
    for (const auto &shape : options::elementwise_correctness_shapes)
    {
        runElementwiseCorrectness(
            shape,
            -8.0,
            8.0,
            33,
            miopenActivationTANH,
            kl::tanh_rocm_float32,
            options::elementwise_float32_absolute_tolerance,
            options::elementwise_float32_relative_tolerance);
    }
}

TEST(ActivationForwardRocmFloat32, SoftmaxCorrectness)
{
    for (const auto &shape : options::softmax_correctness_shapes)
    {
        runSoftmaxCorrectness(
            shape);
    }
}

TEST(ActivationForwardRocmFloat32, ReluBenchmark)
{
    for (const auto &shape : options::elementwise_benchmark_shapes)
    {
        runElementwiseBenchmark(
            "ReLU Forward ROCm Float32",
            shape,
            -10.0,
            10.0,
            11,
            miopenActivationRELU,
            kl::relu_rocm_float32);
    }
}

TEST(ActivationForwardRocmFloat32, SigmoidBenchmark)
{
    for (const auto &shape : options::elementwise_benchmark_shapes)
    {
        runElementwiseBenchmark(
            "Sigmoid Forward ROCm Float32",
            shape,
            -8.0,
            8.0,
            22,
            miopenActivationLOGISTIC,
            kl::sigmoid_rocm_float32);
    }
}

TEST(ActivationForwardRocmFloat32, TanhBenchmark)
{
    for (const auto &shape : options::elementwise_benchmark_shapes)
    {
        runElementwiseBenchmark(
            "Tanh Forward ROCm Float32",
            shape,
            -8.0,
            8.0,
            33,
            miopenActivationTANH,
            kl::tanh_rocm_float32);
    }
}

TEST(ActivationForwardRocmFloat32, SoftmaxBenchmark)
{
    for (const auto &shape : options::softmax_benchmark_shapes)
    {
        runSoftmaxBenchmark(
            "Softmax Forward ROCm Float32",
            shape);
    }
}

#endif // KL_ENABLE_ROCM