#include <gtest/gtest.h>

#ifdef KL_ENABLE_CUDA

#include "common/benchmark_report.hpp"
#include "common/benchmark_timer.hpp"
#include "common/tensor_compare.hpp"
#include "common/tensor_factory.hpp"
#include "common/test_options.hpp"

#include "vendor/cuda/cudnn_activation_backward.cuh"

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>

#include <kernels/cuda/activation/backward_relu_cuda_float32.cuh>
#include <kernels/cuda/activation/backward_sigmoid_cuda_float32.cuh>
#include <kernels/cuda/activation/backward_softmax_cuda_float32.cuh>
#include <kernels/cuda/activation/backward_tanh_cuda_float32.cuh>

#include <kernels/cuda/activation/relu_cuda_float32.cuh>
#include <kernels/cuda/activation/sigmoid_cuda_float32.cuh>
#include <kernels/cuda/activation/softmax_cuda_float32.cuh>
#include <kernels/cuda/activation/tanh_cuda_float32.cuh>

#include <cudnn.h>

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
        cudnnActivationMode_t cudnn_mode,
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
                kl::Device::cuda(),
                input_lower,
                input_upper,
                input_seed);

        const auto cudnn_input =
            kl::test::makeRandomTensor(
                kl::Shape{
                    1,
                    1,
                    1,
                    element_count},
                kl::DType::Float32,
                kl::Device::cuda(),
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
                kl::Device::cuda(),
                -1.0,
                1.0,
                gradient_seed);

        const auto cudnn_gradient =
            kl::test::makeRandomTensor(
                kl::Shape{
                    1,
                    1,
                    1,
                    element_count},
                kl::DType::Float32,
                kl::Device::cuda(),
                -1.0,
                1.0,
                gradient_seed);

        forward_kernel(
            actual_activation_output);

        const auto cudnn_activation_output =
            kl::test::cudnnActivationForwardFloat32(
                cudnn_input,
                cudnn_mode);

        backward_kernel(
            actual_activation_output,
            actual_gradient);

        const auto expected =
            kl::test::cudnnActivationBackwardFloat32(
                cudnn_input,
                cudnn_activation_output,
                cudnn_gradient,
                cudnn_mode);

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
                kl::Device::cuda(),
                -10.0,
                10.0,
                44);

        const auto cudnn_input =
            kl::test::makeRandomTensor(
                kl::Shape{
                    batch_size,
                    class_count,
                    1,
                    1},
                kl::DType::Float32,
                kl::Device::cuda(),
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
                kl::Device::cuda(),
                -1.0,
                1.0,
                55);

        const auto cudnn_gradient =
            kl::test::makeRandomTensor(
                kl::Shape{
                    batch_size,
                    class_count,
                    1,
                    1},
                kl::DType::Float32,
                kl::Device::cuda(),
                -1.0,
                1.0,
                55);

        kl::softmax_cuda_float32(
            actual_activation_output);

        const auto cudnn_activation_output =
            kl::test::cudnnSoftmaxForwardFloat32(
                cudnn_input);

        kl::backward_softmax_cuda_float32(
            actual_activation_output,
            actual_gradient);

        const auto expected =
            kl::test::cudnnSoftmaxBackwardFloat32(
                cudnn_activation_output,
                cudnn_gradient);

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
        cudnnActivationMode_t cudnn_mode,
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
                kl::Device::cuda(),
                input_lower,
                input_upper,
                input_seed);

        const auto cudnn_input =
            kl::test::makeRandomTensor(
                kl::Shape{
                    1,
                    1,
                    1,
                    element_count},
                kl::DType::Float32,
                kl::Device::cuda(),
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
                kl::Device::cuda(),
                -1.0,
                1.0,
                gradient_seed);

        const auto cudnn_gradient =
            kl::test::makeRandomTensor(
                kl::Shape{
                    1,
                    1,
                    1,
                    element_count},
                kl::DType::Float32,
                kl::Device::cuda(),
                -1.0,
                1.0,
                gradient_seed);

        kl::Tensor cudnn_output_gradient(
            kl::Shape{
                1,
                1,
                1,
                element_count},
            kl::DType::Float32,
            kl::Device::cuda());

        forward_kernel(
            kerneloom_activation_output);

        const auto cudnn_activation_output =
            kl::test::cudnnActivationForwardFloat32(
                cudnn_input,
                cudnn_mode);

        kl::test::CudnnHandle cudnn_handle;

        kl::test::CudnnTensorDescriptor tensor_descriptor(
            cudnn_input);

        kl::test::CudnnActivationDescriptor activation_descriptor(
            cudnn_mode);

        const double kerneloom_ms =
            kl::test::benchmarkGpu(
                kl::Device::cuda(),
                options::warmup_iterations,
                options::measured_iterations,
                kl::test::options::benchmark::print_each_iteration,
                "Kerneloom CUDA",
                [&]()
                {
                    backward_kernel(
                        kerneloom_activation_output,
                        kerneloom_gradient);
                });

        const double cudnn_ms =
            kl::test::benchmarkGpu(
                kl::Device::cuda(),
                options::warmup_iterations,
                options::measured_iterations,
                kl::test::options::benchmark::print_each_iteration,
                "cuDNN",
                [&]()
                {
                    kl::test::cudnnActivationBackwardFloat32(
                        cudnn_handle,
                        tensor_descriptor,
                        activation_descriptor,
                        cudnn_activation_output,
                        cudnn_gradient,
                        cudnn_input,
                        cudnn_output_gradient);
                });

        runElementwiseCorrectness(
            element_count,
            input_lower,
            input_upper,
            input_seed,
            gradient_seed,
            cudnn_mode,
            forward_kernel,
            backward_kernel,
            options::elementwise_absolute_tolerance,
            options::elementwise_relative_tolerance);

        kl::test::printBenchmarkComparison(
            benchmark_name,
            "Kerneloom CUDA",
            kerneloom_ms,
            "cuDNN",
            cudnn_ms);
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
                kl::Device::cuda(),
                -10.0,
                10.0,
                44);

        const auto cudnn_input =
            kl::test::makeRandomTensor(
                kl::Shape{
                    batch_size,
                    class_count,
                    1,
                    1},
                kl::DType::Float32,
                kl::Device::cuda(),
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
                kl::Device::cuda(),
                -1.0,
                1.0,
                55);

        const auto cudnn_gradient =
            kl::test::makeRandomTensor(
                kl::Shape{
                    batch_size,
                    class_count,
                    1,
                    1},
                kl::DType::Float32,
                kl::Device::cuda(),
                -1.0,
                1.0,
                55);

        kl::Tensor cudnn_output_gradient(
            kl::Shape{
                batch_size,
                class_count,
                1,
                1},
            kl::DType::Float32,
            kl::Device::cuda());

        kl::softmax_cuda_float32(
            kerneloom_activation_output);

        const auto cudnn_activation_output =
            kl::test::cudnnSoftmaxForwardFloat32(
                cudnn_input);

        kl::test::CudnnHandle cudnn_handle;

        kl::test::CudnnTensorDescriptor tensor_descriptor(
            cudnn_input);

        const double kerneloom_ms =
            kl::test::benchmarkGpu(
                kl::Device::cuda(),
                options::warmup_iterations,
                options::measured_iterations,
                kl::test::options::benchmark::print_each_iteration,
                "Kerneloom CUDA",
                [&]()
                {
                    kl::backward_softmax_cuda_float32(
                        kerneloom_activation_output,
                        kerneloom_gradient);
                });

        const double cudnn_ms =
            kl::test::benchmarkGpu(
                kl::Device::cuda(),
                options::warmup_iterations,
                options::measured_iterations,
                kl::test::options::benchmark::print_each_iteration,
                "cuDNN",
                [&]()
                {
                    kl::test::cudnnSoftmaxBackwardFloat32(
                        cudnn_handle,
                        tensor_descriptor,
                        cudnn_activation_output,
                        cudnn_gradient,
                        cudnn_output_gradient);
                });

        runSoftmaxCorrectness(
            batch_size,
            class_count);

        kl::test::printBenchmarkComparison(
            benchmark_name,
            "Kerneloom CUDA",
            kerneloom_ms,
            "cuDNN",
            cudnn_ms);
    }

}

TEST(ActivationBackwardCudaFloat32, ReluCorrectness_HugeTensor)
{
    runElementwiseCorrectness(
        options::elementwise_huge::element_count,
        -10.0,
        10.0,
        11,
        66,
        CUDNN_ACTIVATION_RELU,
        kl::relu_cuda_float32,
        kl::backward_relu_cuda_float32,
        options::elementwise_absolute_tolerance,
        options::elementwise_relative_tolerance);
}

TEST(ActivationBackwardCudaFloat32, SigmoidCorrectness_HugeTensor)
{
    runElementwiseCorrectness(
        options::elementwise_huge::element_count,
        -8.0,
        8.0,
        22,
        77,
        CUDNN_ACTIVATION_SIGMOID,
        kl::sigmoid_cuda_float32,
        kl::backward_sigmoid_cuda_float32,
        options::elementwise_absolute_tolerance,
        options::elementwise_relative_tolerance);
}

TEST(ActivationBackwardCudaFloat32, TanhCorrectness_HugeTensor)
{
    runElementwiseCorrectness(
        options::elementwise_huge::element_count,
        -8.0,
        8.0,
        33,
        88,
        CUDNN_ACTIVATION_TANH,
        kl::tanh_cuda_float32,
        kl::backward_tanh_cuda_float32,
        options::elementwise_absolute_tolerance,
        options::elementwise_relative_tolerance);
}

TEST(ActivationBackwardCudaFloat32, SoftmaxCorrectness_HugeTensor)
{
    runSoftmaxCorrectness(
        options::softmax_huge::batch_size,
        options::softmax_huge::class_count);
}

TEST(ActivationBackwardCudaFloat32, ReluBenchmark_HugeTensor)
{
    runElementwiseBenchmark(
        "ReLU Backward CUDA Float32 Huge Tensor",
        options::elementwise_odd_huge::element_count,
        -10.0,
        10.0,
        11,
        66,
        CUDNN_ACTIVATION_RELU,
        kl::relu_cuda_float32,
        kl::backward_relu_cuda_float32);
}

TEST(ActivationBackwardCudaFloat32, SigmoidBenchmark_HugeTensor)
{
    runElementwiseBenchmark(
        "Sigmoid Backward CUDA Float32 Huge Tensor",
        options::elementwise_odd_huge::element_count,
        -8.0,
        8.0,
        22,
        77,
        CUDNN_ACTIVATION_SIGMOID,
        kl::sigmoid_cuda_float32,
        kl::backward_sigmoid_cuda_float32);
}

TEST(ActivationBackwardCudaFloat32, TanhBenchmark_HugeTensor)
{
    runElementwiseBenchmark(
        "Tanh Backward CUDA Float32 Huge Tensor",
        options::elementwise_odd_huge::element_count,
        -8.0,
        8.0,
        33,
        88,
        CUDNN_ACTIVATION_TANH,
        kl::tanh_cuda_float32,
        kl::backward_tanh_cuda_float32);
}

TEST(ActivationBackwardCudaFloat32, SoftmaxBenchmark_HugeTensor)
{
    runSoftmaxBenchmark(
        "Softmax Backward CUDA Float32 Huge Tensor",
        options::softmax_odd_huge::batch_size,
        options::softmax_odd_huge::class_count);
}

#endif // KL_ENABLE_CUDA