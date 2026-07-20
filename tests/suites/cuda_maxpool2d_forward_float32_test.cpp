#include <gtest/gtest.h>

#ifdef KL_ENABLE_CUDA

#include "common/benchmark_report.hpp"
#include "common/benchmark_timer.hpp"
#include "common/tensor_compare.hpp"
#include "common/tensor_factory.hpp"
#include "common/test_options.hpp"

#include "vendor/cuda/cudnn_maxpool2d.cuh"

#include <cnn/options/pooling2d_options.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>

#include <kernels/cuda/pooling/maxpool2d_cuda_float32.cuh>

namespace
{

    namespace options =
        kl::test::options::maxpool2d_forward_float32;

    kl::Pooling2dOptions make_pooling_options()
    {
        kl::Pooling2dOptions pooling_options;

        pooling_options.kernel_h =
            options::kernel_height;

        pooling_options.kernel_w =
            options::kernel_width;

        pooling_options.stride_h =
            options::stride_height;

        pooling_options.stride_w =
            options::stride_width;

        pooling_options.padding_h =
            options::padding_height;

        pooling_options.padding_w =
            options::padding_width;

        return pooling_options;
    }

}

TEST(MaxPool2dForwardCudaFloat32, MatchesCuDNN)
{
    const auto input = kl::test::makeRandomTensor(
        kl::Shape{
            options::batch_size,
            options::channels,
            options::input_height,
            options::input_width},
        kl::DType::Float32,
        kl::Device::cuda(),
        -1.0,
        1.0,
        11);

    kl::Tensor actual(
        kl::Shape{
            options::batch_size,
            options::channels,
            options::output_height,
            options::output_width},
        kl::DType::Float32,
        kl::Device::cuda());

    const kl::Pooling2dOptions pooling_options =
        make_pooling_options();

    kl::maxpool2d_cuda_float32(
        input,
        actual,
        pooling_options);

    const auto expected =
        kl::test::cudnnMaxPool2dForwardFloat32(
            input,
            options::kernel_height,
            options::kernel_width,
            options::padding_height,
            options::padding_width,
            options::stride_height,
            options::stride_width);

    EXPECT_TRUE(kl::test::tensorCompare(
        expected,
        actual,
        options::absolute_tolerance,
        options::relative_tolerance));
}

TEST(MaxPool2dForwardCudaFloat32, BenchmarkAgainstCuDNN)
{
    const auto input = kl::test::makeRandomTensor(
        kl::Shape{
            options::batch_size,
            options::channels,
            options::input_height,
            options::input_width},
        kl::DType::Float32,
        kl::Device::cuda(),
        -1.0,
        1.0,
        11);

    kl::Tensor kerneloom_output(
        kl::Shape{
            options::batch_size,
            options::channels,
            options::output_height,
            options::output_width},
        kl::DType::Float32,
        kl::Device::cuda());

    kl::Tensor cudnn_output(
        kl::Shape{
            options::batch_size,
            options::channels,
            options::output_height,
            options::output_width},
        kl::DType::Float32,
        kl::Device::cuda());

    const kl::Pooling2dOptions pooling_options =
        make_pooling_options();

    kl::test::CudnnHandle cudnn_handle;

    kl::test::CudnnTensorDescriptor input_descriptor(
        input);

    kl::test::CudnnTensorDescriptor cudnn_output_descriptor(
        cudnn_output);

    kl::test::CudnnPoolingDescriptor cudnn_pooling_descriptor(
        options::kernel_height,
        options::kernel_width,
        options::padding_height,
        options::padding_width,
        options::stride_height,
        options::stride_width);

    const double kerneloom_ms = kl::test::benchmarkGpu(
        kl::Device::cuda(),
        options::warmup_iterations,
        options::measured_iterations,
        options::print_each_iteration,
        "Kerneloom CUDA",
        [&]()
        {
            kl::maxpool2d_cuda_float32(
                input,
                kerneloom_output,
                pooling_options);
        });

    const double cudnn_ms = kl::test::benchmarkGpu(
        kl::Device::cuda(),
        options::warmup_iterations,
        options::measured_iterations,
        options::print_each_iteration,
        "cuDNN",
        [&]()
        {
            kl::test::cudnnMaxPool2dForwardFloat32(
                cudnn_handle,
                input_descriptor,
                cudnn_output_descriptor,
                cudnn_pooling_descriptor,
                input,
                cudnn_output);
        });

    EXPECT_TRUE(kl::test::tensorCompare(
        cudnn_output,
        kerneloom_output,
        options::absolute_tolerance,
        options::relative_tolerance));

    kl::test::printBenchmarkComparison(
        "MaxPool2d Forward CUDA Float32",
        "Kerneloom CUDA",
        kerneloom_ms,
        "cuDNN",
        cudnn_ms);
}

#endif // KL_ENABLE_CUDA