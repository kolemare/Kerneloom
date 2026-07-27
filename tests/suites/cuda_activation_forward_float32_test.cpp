#include <gtest/gtest.h>

#ifdef KL_ENABLE_CUDA

#include "common/benchmark_report.hpp"
#include "common/benchmark_timer.hpp"
#include "common/tensor_compare.hpp"
#include "common/tensor_factory.hpp"
#include "common/test_options.hpp"

#include "options/activation_forward_options.hpp"

#include "vendor/cuda/cudnn_activation.cuh"

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>

#include <kernels/cuda/activation/relu_cuda_float32.cuh>
#include <kernels/cuda/activation/sigmoid_cuda_float32.cuh>
#include <kernels/cuda/activation/softmax_cuda_float32.cuh>
#include <kernels/cuda/activation/tanh_cuda_float32.cuh>

#include <cudnn.h>

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
        cudnnActivationMode_t cudnn_mode,
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
                kl::Device::cuda(),
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
                kl::Device::cuda(),
                lower,
                upper,
                seed);

        kl::test::CudnnHandle cudnn_handle;

        kl::test::CudnnTensorDescriptor tensor_descriptor(
            expected);

        kl::test::CudnnActivationDescriptor activation_descriptor(
            cudnn_mode);

        kernel(
            actual);

        kl::test::cudnnActivationForwardFloat32(
            cudnn_handle,
            tensor_descriptor,
            activation_descriptor,
            expected,
            expected);

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
                kl::Device::cuda(),
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
                kl::Device::cuda(),
                -10.0,
                10.0,
                44);

        kl::test::CudnnHandle cudnn_handle;

        kl::test::CudnnTensorDescriptor tensor_descriptor(
            expected);

        kl::softmax_cuda_float32(
            actual);

        kl::test::cudnnSoftmaxForwardFloat32(
            cudnn_handle,
            tensor_descriptor,
            expected,
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
        cudnnActivationMode_t cudnn_mode,
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
                kl::Device::cuda(),
                lower,
                upper,
                seed);

        auto cudnn_tensor =
            kl::test::makeRandomTensor(
                kl::Shape{
                    1,
                    1,
                    1,
                    shape.element_count},
                kl::DType::Float32,
                kl::Device::cuda(),
                lower,
                upper,
                seed);

        kl::test::CudnnHandle cudnn_handle;

        kl::test::CudnnTensorDescriptor cudnn_tensor_descriptor(
            cudnn_tensor);

        kl::test::CudnnActivationDescriptor cudnn_activation_descriptor(
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
                    kernel(
                        kerneloom_tensor);
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
                    kl::test::cudnnActivationForwardFloat32(
                        cudnn_handle,
                        cudnn_tensor_descriptor,
                        cudnn_activation_descriptor,
                        cudnn_tensor,
                        cudnn_tensor);
                });

        EXPECT_TRUE(kl::test::tensorCompare(
            cudnn_tensor,
            kerneloom_tensor,
            options::elementwise_float32_absolute_tolerance,
            options::elementwise_float32_relative_tolerance));

        const std::string full_benchmark_name =
            std::string(benchmark_name) +
            " " +
            shape.name;

        kl::test::printBenchmarkComparison(
            full_benchmark_name.c_str(),
            "Kerneloom CUDA",
            kerneloom_ms,
            "cuDNN",
            cudnn_ms);
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
                kl::Device::cuda(),
                -10.0,
                10.0,
                44);

        auto cudnn_tensor =
            kl::test::makeRandomTensor(
                kl::Shape{
                    shape.batch_size,
                    shape.class_count,
                    1,
                    1},
                kl::DType::Float32,
                kl::Device::cuda(),
                -10.0,
                10.0,
                44);

        kl::test::CudnnHandle cudnn_handle;

        kl::test::CudnnTensorDescriptor cudnn_tensor_descriptor(
            cudnn_tensor);

        const double kerneloom_ms =
            kl::test::benchmarkGpu(
                kl::Device::cuda(),
                options::warmup_iterations,
                options::measured_iterations,
                kl::test::options::benchmark::print_each_iteration,
                "Kerneloom CUDA",
                [&]()
                {
                    kl::softmax_cuda_float32(
                        kerneloom_tensor);
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
                    kl::test::cudnnSoftmaxForwardFloat32(
                        cudnn_handle,
                        cudnn_tensor_descriptor,
                        cudnn_tensor,
                        cudnn_tensor);
                });

        EXPECT_TRUE(kl::test::tensorCompare(
            cudnn_tensor,
            kerneloom_tensor,
            options::softmax_float32_absolute_tolerance,
            options::softmax_float32_relative_tolerance));

        const std::string full_benchmark_name =
            std::string(benchmark_name) +
            " " +
            shape.name;

        kl::test::printBenchmarkComparison(
            full_benchmark_name.c_str(),
            "Kerneloom CUDA",
            kerneloom_ms,
            "cuDNN",
            cudnn_ms);
    }

}

TEST(ActivationForwardCudaFloat32, ReluCorrectness)
{
    for (const auto &shape : options::elementwise_correctness_shapes)
    {
        runElementwiseCorrectness(
            shape,
            -10.0,
            10.0,
            11,
            CUDNN_ACTIVATION_RELU,
            kl::relu_cuda_float32,
            options::elementwise_float32_absolute_tolerance,
            options::elementwise_float32_relative_tolerance);
    }
}

TEST(ActivationForwardCudaFloat32, SigmoidCorrectness)
{
    for (const auto &shape : options::elementwise_correctness_shapes)
    {
        runElementwiseCorrectness(
            shape,
            -8.0,
            8.0,
            22,
            CUDNN_ACTIVATION_SIGMOID,
            kl::sigmoid_cuda_float32,
            options::elementwise_float32_absolute_tolerance,
            options::elementwise_float32_relative_tolerance);
    }
}

TEST(ActivationForwardCudaFloat32, TanhCorrectness)
{
    for (const auto &shape : options::elementwise_correctness_shapes)
    {
        runElementwiseCorrectness(
            shape,
            -8.0,
            8.0,
            33,
            CUDNN_ACTIVATION_TANH,
            kl::tanh_cuda_float32,
            options::elementwise_float32_absolute_tolerance,
            options::elementwise_float32_relative_tolerance);
    }
}

TEST(ActivationForwardCudaFloat32, SoftmaxCorrectness)
{
    for (const auto &shape : options::softmax_correctness_shapes)
    {
        runSoftmaxCorrectness(
            shape);
    }
}

TEST(ActivationForwardCudaFloat32, ReluBenchmark)
{
    for (const auto &shape : options::elementwise_benchmark_shapes)
    {
        runElementwiseBenchmark(
            "ReLU Forward CUDA Float32",
            shape,
            -10.0,
            10.0,
            11,
            CUDNN_ACTIVATION_RELU,
            kl::relu_cuda_float32);
    }
}

TEST(ActivationForwardCudaFloat32, SigmoidBenchmark)
{
    for (const auto &shape : options::elementwise_benchmark_shapes)
    {
        runElementwiseBenchmark(
            "Sigmoid Forward CUDA Float32",
            shape,
            -8.0,
            8.0,
            22,
            CUDNN_ACTIVATION_SIGMOID,
            kl::sigmoid_cuda_float32);
    }
}

TEST(ActivationForwardCudaFloat32, TanhBenchmark)
{
    for (const auto &shape : options::elementwise_benchmark_shapes)
    {
        runElementwiseBenchmark(
            "Tanh Forward CUDA Float32",
            shape,
            -8.0,
            8.0,
            33,
            CUDNN_ACTIVATION_TANH,
            kl::tanh_cuda_float32);
    }
}

TEST(ActivationForwardCudaFloat32, SoftmaxBenchmark)
{
    for (const auto &shape : options::softmax_benchmark_shapes)
    {
        runSoftmaxBenchmark(
            "Softmax Forward CUDA Float32",
            shape);
    }
}

#endif // KL_ENABLE_CUDA