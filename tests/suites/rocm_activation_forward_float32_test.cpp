#include <gtest/gtest.h>

#ifdef KL_ENABLE_ROCM

#include "common/benchmark_report.hpp"
#include "common/benchmark_timer.hpp"
#include "common/tensor_compare.hpp"
#include "common/tensor_factory.hpp"
#include "common/test_options.hpp"

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

namespace
{

    namespace options =
        kl::test::options::activation_forward_float32;

    using ActivationKernel =
        void (*)(kl::Tensor &);

    void runElementwiseCorrectness(
        std::size_t element_count,
        double lower,
        double upper,
        int seed,
        miopenActivationMode_t miopen_mode,
        ActivationKernel kernel,
        double absolute_tolerance,
        double relative_tolerance)
    {
        auto actual =
            kl::test::makeRandomTensor(
                kl::Shape{
                    1,
                    1,
                    1,
                    element_count},
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
                    element_count},
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
        std::size_t batch_size,
        std::size_t class_count)
    {
        auto actual =
            kl::test::makeRandomTensor(
                kl::Shape{
                    batch_size,
                    class_count,
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
                    batch_size,
                    class_count,
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
            options::softmax_absolute_tolerance,
            options::softmax_relative_tolerance));
    }

    void runElementwiseBenchmark(
        const char *benchmark_name,
        std::size_t element_count,
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
                    element_count},
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
                    element_count},
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
            options::elementwise_absolute_tolerance,
            options::elementwise_relative_tolerance));

        kl::test::printBenchmarkComparison(
            benchmark_name,
            "Kerneloom ROCm",
            kerneloom_ms,
            "MIOpen",
            miopen_ms);
    }

    void runSoftmaxBenchmark(
        const char *benchmark_name,
        std::size_t batch_size,
        std::size_t class_count)
    {
        auto kerneloom_tensor =
            kl::test::makeRandomTensor(
                kl::Shape{
                    batch_size,
                    class_count,
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
                    batch_size,
                    class_count,
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
            options::softmax_absolute_tolerance,
            options::softmax_relative_tolerance));

        kl::test::printBenchmarkComparison(
            benchmark_name,
            "Kerneloom ROCm",
            kerneloom_ms,
            "MIOpen",
            miopen_ms);
    }

}

TEST(ActivationForwardRocmFloat32, ReluCorrectness_HugeTensor)
{
    runElementwiseCorrectness(
        options::elementwise_huge::element_count,
        -10.0,
        10.0,
        11,
        miopenActivationRELU,
        kl::relu_rocm_float32,
        options::elementwise_absolute_tolerance,
        options::elementwise_relative_tolerance);
}

TEST(ActivationForwardRocmFloat32, SigmoidCorrectness_HugeTensor)
{
    runElementwiseCorrectness(
        options::elementwise_huge::element_count,
        -8.0,
        8.0,
        22,
        miopenActivationLOGISTIC,
        kl::sigmoid_rocm_float32,
        options::elementwise_absolute_tolerance,
        options::elementwise_relative_tolerance);
}

TEST(ActivationForwardRocmFloat32, TanhCorrectness_HugeTensor)
{
    runElementwiseCorrectness(
        options::elementwise_huge::element_count,
        -8.0,
        8.0,
        33,
        miopenActivationTANH,
        kl::tanh_rocm_float32,
        options::elementwise_absolute_tolerance,
        options::elementwise_relative_tolerance);
}

TEST(ActivationForwardRocmFloat32, SoftmaxCorrectness_HugeTensor)
{
    runSoftmaxCorrectness(
        options::softmax_huge::batch_size,
        options::softmax_huge::class_count);
}

TEST(ActivationForwardRocmFloat32, ReluBenchmark_HugeTensor)
{
    runElementwiseBenchmark(
        "ReLU Forward ROCm Float32 Huge Tensor",
        options::elementwise_odd_huge::element_count,
        -10.0,
        10.0,
        11,
        miopenActivationRELU,
        kl::relu_rocm_float32);
}

TEST(ActivationForwardRocmFloat32, SigmoidBenchmark_HugeTensor)
{
    runElementwiseBenchmark(
        "Sigmoid Forward ROCm Float32 Huge Tensor",
        options::elementwise_odd_huge::element_count,
        -8.0,
        8.0,
        22,
        miopenActivationLOGISTIC,
        kl::sigmoid_rocm_float32);
}

TEST(ActivationForwardRocmFloat32, TanhBenchmark_HugeTensor)
{
    runElementwiseBenchmark(
        "Tanh Forward ROCm Float32 Huge Tensor",
        options::elementwise_odd_huge::element_count,
        -8.0,
        8.0,
        33,
        miopenActivationTANH,
        kl::tanh_rocm_float32);
}

TEST(ActivationForwardRocmFloat32, SoftmaxBenchmark_HugeTensor)
{
    runSoftmaxBenchmark(
        "Softmax Forward ROCm Float32 Huge Tensor",
        options::softmax_odd_huge::batch_size,
        options::softmax_odd_huge::class_count);
}

#endif // KL_ENABLE_ROCM