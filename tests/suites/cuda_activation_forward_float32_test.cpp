#include <gtest/gtest.h>

#ifdef KL_ENABLE_CUDA

#include "common/benchmark_report.hpp"
#include "common/benchmark_timer.hpp"
#include "common/tensor_compare.hpp"
#include "common/tensor_factory.hpp"
#include "common/test_options.hpp"

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
        cudnnActivationMode_t cudnn_mode,
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
                    element_count},
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
                kl::Device::cuda(),
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
            options::softmax_absolute_tolerance,
            options::softmax_relative_tolerance));
    }

    void runElementwiseBenchmark(
        const char *benchmark_name,
        std::size_t element_count,
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
                    element_count},
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
                    element_count},
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
            options::elementwise_absolute_tolerance,
            options::elementwise_relative_tolerance));

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
        auto kerneloom_tensor =
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

        auto cudnn_tensor =
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
            options::softmax_absolute_tolerance,
            options::softmax_relative_tolerance));

        kl::test::printBenchmarkComparison(
            benchmark_name,
            "Kerneloom CUDA",
            kerneloom_ms,
            "cuDNN",
            cudnn_ms);
    }

}

TEST(ActivationForwardCudaFloat32, ReluCorrectness_HugeTensor)
{
    runElementwiseCorrectness(
        options::elementwise_huge::element_count,
        -10.0,
        10.0,
        11,
        CUDNN_ACTIVATION_RELU,
        kl::relu_cuda_float32,
        options::elementwise_absolute_tolerance,
        options::elementwise_relative_tolerance);
}

TEST(ActivationForwardCudaFloat32, SigmoidCorrectness_HugeTensor)
{
    runElementwiseCorrectness(
        options::elementwise_huge::element_count,
        -8.0,
        8.0,
        22,
        CUDNN_ACTIVATION_SIGMOID,
        kl::sigmoid_cuda_float32,
        options::elementwise_absolute_tolerance,
        options::elementwise_relative_tolerance);
}

TEST(ActivationForwardCudaFloat32, TanhCorrectness_HugeTensor)
{
    runElementwiseCorrectness(
        options::elementwise_huge::element_count,
        -8.0,
        8.0,
        33,
        CUDNN_ACTIVATION_TANH,
        kl::tanh_cuda_float32,
        options::elementwise_absolute_tolerance,
        options::elementwise_relative_tolerance);
}

TEST(ActivationForwardCudaFloat32, SoftmaxCorrectness_HugeTensor)
{
    runSoftmaxCorrectness(
        options::softmax_huge::batch_size,
        options::softmax_huge::class_count);
}

TEST(ActivationForwardCudaFloat32, ReluBenchmark_HugeTensor)
{
    runElementwiseBenchmark(
        "ReLU Forward CUDA Float32 Huge Tensor",
        options::elementwise_odd_huge::element_count,
        -10.0,
        10.0,
        11,
        CUDNN_ACTIVATION_RELU,
        kl::relu_cuda_float32);
}

TEST(ActivationForwardCudaFloat32, SigmoidBenchmark_HugeTensor)
{
    runElementwiseBenchmark(
        "Sigmoid Forward CUDA Float32 Huge Tensor",
        options::elementwise_odd_huge::element_count,
        -8.0,
        8.0,
        22,
        CUDNN_ACTIVATION_SIGMOID,
        kl::sigmoid_cuda_float32);
}

TEST(ActivationForwardCudaFloat32, TanhBenchmark_HugeTensor)
{
    runElementwiseBenchmark(
        "Tanh Forward CUDA Float32 Huge Tensor",
        options::elementwise_odd_huge::element_count,
        -8.0,
        8.0,
        33,
        CUDNN_ACTIVATION_TANH,
        kl::tanh_cuda_float32);
}

TEST(ActivationForwardCudaFloat32, SoftmaxBenchmark_HugeTensor)
{
    runSoftmaxBenchmark(
        "Softmax Forward CUDA Float32 Huge Tensor",
        options::softmax_odd_huge::batch_size,
        options::softmax_odd_huge::class_count);
}

#endif // KL_ENABLE_CUDA