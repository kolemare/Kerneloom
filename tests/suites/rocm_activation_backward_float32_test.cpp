#include <gtest/gtest.h>

#ifdef KL_ENABLE_ROCM

#include "common/benchmark_report.hpp"
#include "common/benchmark_timer.hpp"
#include "common/tensor_compare.hpp"
#include "common/tensor_factory.hpp"
#include "common/test_options.hpp"

#include "vendor/rocm/miopen_activation_backward.hiph"

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>

#include <kernels/rocm/activation/backward_relu_rocm_float32.hiph>
#include <kernels/rocm/activation/backward_sigmoid_rocm_float32.hiph>
#include <kernels/rocm/activation/backward_softmax_rocm_float32.hiph>
#include <kernels/rocm/activation/backward_tanh_rocm_float32.hiph>

#include <kernels/rocm/activation/relu_rocm_float32.hiph>
#include <kernels/rocm/activation/sigmoid_rocm_float32.hiph>
#include <kernels/rocm/activation/softmax_rocm_float32.hiph>
#include <kernels/rocm/activation/tanh_rocm_float32.hiph>

#include <miopen/miopen.h>

#include <cstddef>

namespace
{

    namespace options =
        kl::test::options::activation_backward_float32;

    using ForwardKernel =
        void (*)(kl::Tensor &);

    using BackwardKernel =
        void (*)(const kl::Tensor &, kl::Tensor &);

    void runElementwiseCorrectness(
        std::size_t element_count,
        double input_lower,
        double input_upper,
        int input_seed,
        int gradient_seed,
        miopenActivationMode_t miopen_mode,
        ForwardKernel forward_kernel,
        BackwardKernel backward_kernel,
        double absolute_tolerance,
        double relative_tolerance)
    {
        auto actual_activation_output =
            kl::test::makeRandomTensor(
                kl::Shape{
                    1,
                    1,
                    1,
                    element_count},
                kl::DType::Float32,
                kl::Device::rocm(),
                input_lower,
                input_upper,
                input_seed);

        const auto miopen_input =
            kl::test::makeRandomTensor(
                kl::Shape{
                    1,
                    1,
                    1,
                    element_count},
                kl::DType::Float32,
                kl::Device::rocm(),
                input_lower,
                input_upper,
                input_seed);

        auto actual_gradient =
            kl::test::makeRandomTensor(
                kl::Shape{
                    1,
                    1,
                    1,
                    element_count},
                kl::DType::Float32,
                kl::Device::rocm(),
                -1.0,
                1.0,
                gradient_seed);

        const auto miopen_gradient =
            kl::test::makeRandomTensor(
                kl::Shape{
                    1,
                    1,
                    1,
                    element_count},
                kl::DType::Float32,
                kl::Device::rocm(),
                -1.0,
                1.0,
                gradient_seed);

        forward_kernel(
            actual_activation_output);

        const auto miopen_activation_output =
            kl::test::miopenActivationForwardFloat32(
                miopen_input,
                miopen_mode);

        backward_kernel(
            actual_activation_output,
            actual_gradient);

        const auto expected =
            kl::test::miopenActivationBackwardFloat32(
                miopen_input,
                miopen_activation_output,
                miopen_gradient,
                miopen_mode);

        EXPECT_TRUE(kl::test::tensorCompare(
            expected,
            actual_gradient,
            absolute_tolerance,
            relative_tolerance));
    }

    void runSoftmaxCorrectness(
        std::size_t batch_size,
        std::size_t class_count)
    {
        auto actual_activation_output =
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

        const auto miopen_input =
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

        auto actual_gradient =
            kl::test::makeRandomTensor(
                kl::Shape{
                    batch_size,
                    class_count,
                    1,
                    1},
                kl::DType::Float32,
                kl::Device::rocm(),
                -1.0,
                1.0,
                55);

        const auto miopen_gradient =
            kl::test::makeRandomTensor(
                kl::Shape{
                    batch_size,
                    class_count,
                    1,
                    1},
                kl::DType::Float32,
                kl::Device::rocm(),
                -1.0,
                1.0,
                55);

        kl::softmax_rocm_float32(
            actual_activation_output);

        const auto miopen_activation_output =
            kl::test::miopenSoftmaxForwardFloat32(
                miopen_input);

        kl::backward_softmax_rocm_float32(
            actual_activation_output,
            actual_gradient);

        const auto expected =
            kl::test::miopenSoftmaxBackwardFloat32(
                miopen_activation_output,
                miopen_gradient);

        EXPECT_TRUE(kl::test::tensorCompare(
            expected,
            actual_gradient,
            options::softmax_absolute_tolerance,
            options::softmax_relative_tolerance));
    }

    void runElementwiseBenchmark(
        const char *benchmark_name,
        std::size_t element_count,
        double input_lower,
        double input_upper,
        int input_seed,
        int gradient_seed,
        miopenActivationMode_t miopen_mode,
        ForwardKernel forward_kernel,
        BackwardKernel backward_kernel)
    {
        auto kerneloom_activation_output =
            kl::test::makeRandomTensor(
                kl::Shape{
                    1,
                    1,
                    1,
                    element_count},
                kl::DType::Float32,
                kl::Device::rocm(),
                input_lower,
                input_upper,
                input_seed);

        const auto miopen_input =
            kl::test::makeRandomTensor(
                kl::Shape{
                    1,
                    1,
                    1,
                    element_count},
                kl::DType::Float32,
                kl::Device::rocm(),
                input_lower,
                input_upper,
                input_seed);

        auto kerneloom_gradient =
            kl::test::makeRandomTensor(
                kl::Shape{
                    1,
                    1,
                    1,
                    element_count},
                kl::DType::Float32,
                kl::Device::rocm(),
                -1.0,
                1.0,
                gradient_seed);

        const auto miopen_gradient =
            kl::test::makeRandomTensor(
                kl::Shape{
                    1,
                    1,
                    1,
                    element_count},
                kl::DType::Float32,
                kl::Device::rocm(),
                -1.0,
                1.0,
                gradient_seed);

        kl::Tensor miopen_output_gradient(
            kl::Shape{
                1,
                1,
                1,
                element_count},
            kl::DType::Float32,
            kl::Device::rocm());

        forward_kernel(
            kerneloom_activation_output);

        const auto miopen_activation_output =
            kl::test::miopenActivationForwardFloat32(
                miopen_input,
                miopen_mode);

        kl::test::MiopenHandle miopen_handle;

        kl::test::MiopenTensorDescriptor tensor_descriptor(
            miopen_input);

        kl::test::MiopenActivationDescriptor activation_descriptor(
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
                    backward_kernel(
                        kerneloom_activation_output,
                        kerneloom_gradient);
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
                    kl::test::miopenActivationBackwardFloat32(
                        miopen_handle,
                        tensor_descriptor,
                        activation_descriptor,
                        miopen_activation_output,
                        miopen_gradient,
                        miopen_input,
                        miopen_output_gradient);
                });

        runElementwiseCorrectness(
            element_count,
            input_lower,
            input_upper,
            input_seed,
            gradient_seed,
            miopen_mode,
            forward_kernel,
            backward_kernel,
            options::elementwise_absolute_tolerance,
            options::elementwise_relative_tolerance);

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
        auto kerneloom_activation_output =
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

        const auto miopen_input =
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

        auto kerneloom_gradient =
            kl::test::makeRandomTensor(
                kl::Shape{
                    batch_size,
                    class_count,
                    1,
                    1},
                kl::DType::Float32,
                kl::Device::rocm(),
                -1.0,
                1.0,
                55);

        const auto miopen_gradient =
            kl::test::makeRandomTensor(
                kl::Shape{
                    batch_size,
                    class_count,
                    1,
                    1},
                kl::DType::Float32,
                kl::Device::rocm(),
                -1.0,
                1.0,
                55);

        kl::Tensor miopen_output_gradient(
            kl::Shape{
                batch_size,
                class_count,
                1,
                1},
            kl::DType::Float32,
            kl::Device::rocm());

        kl::softmax_rocm_float32(
            kerneloom_activation_output);

        const auto miopen_activation_output =
            kl::test::miopenSoftmaxForwardFloat32(
                miopen_input);

        kl::test::MiopenHandle miopen_handle;

        kl::test::MiopenTensorDescriptor tensor_descriptor(
            miopen_input);

        const double kerneloom_ms =
            kl::test::benchmarkGpu(
                kl::Device::rocm(),
                options::warmup_iterations,
                options::measured_iterations,
                kl::test::options::benchmark::print_each_iteration,
                "Kerneloom ROCm",
                [&]()
                {
                    kl::backward_softmax_rocm_float32(
                        kerneloom_activation_output,
                        kerneloom_gradient);
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
                    kl::test::miopenSoftmaxBackwardFloat32(
                        miopen_handle,
                        tensor_descriptor,
                        miopen_activation_output,
                        miopen_gradient,
                        miopen_output_gradient);
                });

        runSoftmaxCorrectness(
            batch_size,
            class_count);

        kl::test::printBenchmarkComparison(
            benchmark_name,
            "Kerneloom ROCm",
            kerneloom_ms,
            "MIOpen",
            miopen_ms);
    }

}

TEST(ActivationBackwardRocmFloat32, ReluCorrectness_HugeTensor)
{
    runElementwiseCorrectness(
        options::elementwise_huge::element_count,
        -10.0,
        10.0,
        11,
        66,
        miopenActivationRELU,
        kl::relu_rocm_float32,
        kl::backward_relu_rocm_float32,
        options::elementwise_absolute_tolerance,
        options::elementwise_relative_tolerance);
}

TEST(ActivationBackwardRocmFloat32, SigmoidCorrectness_HugeTensor)
{
    runElementwiseCorrectness(
        options::elementwise_huge::element_count,
        -8.0,
        8.0,
        22,
        77,
        miopenActivationLOGISTIC,
        kl::sigmoid_rocm_float32,
        kl::backward_sigmoid_rocm_float32,
        options::elementwise_absolute_tolerance,
        options::elementwise_relative_tolerance);
}

TEST(ActivationBackwardRocmFloat32, TanhCorrectness_HugeTensor)
{
    runElementwiseCorrectness(
        options::elementwise_huge::element_count,
        -8.0,
        8.0,
        33,
        88,
        miopenActivationTANH,
        kl::tanh_rocm_float32,
        kl::backward_tanh_rocm_float32,
        options::elementwise_absolute_tolerance,
        options::elementwise_relative_tolerance);
}

TEST(ActivationBackwardRocmFloat32, SoftmaxCorrectness_HugeTensor)
{
    runSoftmaxCorrectness(
        options::softmax_huge::batch_size,
        options::softmax_huge::class_count);
}

TEST(ActivationBackwardRocmFloat32, ReluBenchmark_HugeTensor)
{
    runElementwiseBenchmark(
        "ReLU Backward ROCm Float32 Huge Tensor",
        options::elementwise_odd_huge::element_count,
        -10.0,
        10.0,
        11,
        66,
        miopenActivationRELU,
        kl::relu_rocm_float32,
        kl::backward_relu_rocm_float32);
}

TEST(ActivationBackwardRocmFloat32, SigmoidBenchmark_HugeTensor)
{
    runElementwiseBenchmark(
        "Sigmoid Backward ROCm Float32 Huge Tensor",
        options::elementwise_odd_huge::element_count,
        -8.0,
        8.0,
        22,
        77,
        miopenActivationLOGISTIC,
        kl::sigmoid_rocm_float32,
        kl::backward_sigmoid_rocm_float32);
}

TEST(ActivationBackwardRocmFloat32, TanhBenchmark_HugeTensor)
{
    runElementwiseBenchmark(
        "Tanh Backward ROCm Float32 Huge Tensor",
        options::elementwise_odd_huge::element_count,
        -8.0,
        8.0,
        33,
        88,
        miopenActivationTANH,
        kl::tanh_rocm_float32,
        kl::backward_tanh_rocm_float32);
}

TEST(ActivationBackwardRocmFloat32, SoftmaxBenchmark_HugeTensor)
{
    runSoftmaxBenchmark(
        "Softmax Backward ROCm Float32 Huge Tensor",
        options::softmax_odd_huge::batch_size,
        options::softmax_odd_huge::class_count);
}

#endif // KL_ENABLE_ROCM