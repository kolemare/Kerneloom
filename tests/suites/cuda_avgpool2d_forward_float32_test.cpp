#include <gtest/gtest.h>

#ifdef KL_ENABLE_CUDA

#include "common/benchmark_report.hpp"
#include "common/benchmark_timer.hpp"
#include "common/tensor_compare.hpp"
#include "common/tensor_factory.hpp"
#include "common/test_options.hpp"

#include "vendor/cuda/cudnn_avgpool2d.cuh"

#include <cnn/options/pooling2d_options.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>

#include <kernels/cuda/pooling/avgpool2d_cuda_float32.cuh>

#include <cstddef>

namespace
{

    namespace options =
        kl::test::options::avgpool2d_forward_float32;

    kl::Pooling2dOptions make_pooling_options(
        std::size_t kernel_height,
        std::size_t kernel_width,
        std::size_t stride_height,
        std::size_t stride_width,
        std::size_t padding_height,
        std::size_t padding_width)
    {
        kl::Pooling2dOptions pooling_options;

        pooling_options.kernel_h =
            kernel_height;

        pooling_options.kernel_w =
            kernel_width;

        pooling_options.stride_h =
            stride_height;

        pooling_options.stride_w =
            stride_width;

        pooling_options.padding_h =
            padding_height;

        pooling_options.padding_w =
            padding_width;

        return pooling_options;
    }

    void runCorrectness(
        std::size_t batch_size,
        std::size_t channels,
        std::size_t input_height,
        std::size_t input_width,
        std::size_t output_height,
        std::size_t output_width,
        std::size_t kernel_height,
        std::size_t kernel_width,
        std::size_t stride_height,
        std::size_t stride_width,
        std::size_t padding_height,
        std::size_t padding_width)
    {
        const auto input = kl::test::makeRandomTensor(
            kl::Shape{
                batch_size,
                channels,
                input_height,
                input_width},
            kl::DType::Float32,
            kl::Device::cuda(),
            -1.0,
            1.0,
            11);

        kl::Tensor actual(
            kl::Shape{
                batch_size,
                channels,
                output_height,
                output_width},
            kl::DType::Float32,
            kl::Device::cuda());

        const kl::Pooling2dOptions pooling_options =
            make_pooling_options(
                kernel_height,
                kernel_width,
                stride_height,
                stride_width,
                padding_height,
                padding_width);

        kl::avgpool2d_cuda_float32(
            input,
            actual,
            pooling_options);

        const auto expected =
            kl::test::cudnnAvgPool2dForwardFloat32(
                input,
                kernel_height,
                kernel_width,
                padding_height,
                padding_width,
                stride_height,
                stride_width);

        EXPECT_TRUE(kl::test::tensorCompare(
            expected,
            actual,
            options::absolute_tolerance,
            options::relative_tolerance));
    }

    void runBenchmark(
        const char *benchmark_name,
        std::size_t batch_size,
        std::size_t channels,
        std::size_t input_height,
        std::size_t input_width,
        std::size_t output_height,
        std::size_t output_width,
        std::size_t kernel_height,
        std::size_t kernel_width,
        std::size_t stride_height,
        std::size_t stride_width,
        std::size_t padding_height,
        std::size_t padding_width)
    {
        const auto input = kl::test::makeRandomTensor(
            kl::Shape{
                batch_size,
                channels,
                input_height,
                input_width},
            kl::DType::Float32,
            kl::Device::cuda(),
            -1.0,
            1.0,
            11);

        kl::Tensor kerneloom_output(
            kl::Shape{
                batch_size,
                channels,
                output_height,
                output_width},
            kl::DType::Float32,
            kl::Device::cuda());

        kl::Tensor cudnn_output(
            kl::Shape{
                batch_size,
                channels,
                output_height,
                output_width},
            kl::DType::Float32,
            kl::Device::cuda());

        const kl::Pooling2dOptions pooling_options =
            make_pooling_options(
                kernel_height,
                kernel_width,
                stride_height,
                stride_width,
                padding_height,
                padding_width);

        kl::test::CudnnHandle cudnn_handle;

        kl::test::CudnnTensorDescriptor input_descriptor(
            input);

        kl::test::CudnnTensorDescriptor cudnn_output_descriptor(
            cudnn_output);

        kl::test::CudnnAveragePoolingDescriptor cudnn_pooling_descriptor(
            kernel_height,
            kernel_width,
            padding_height,
            padding_width,
            stride_height,
            stride_width);

        const double kerneloom_ms = kl::test::benchmarkGpu(
            kl::Device::cuda(),
            options::warmup_iterations,
            options::measured_iterations,
            kl::test::options::benchmark::print_each_iteration,
            "Kerneloom CUDA",
            [&]()
            {
                kl::avgpool2d_cuda_float32(
                    input,
                    kerneloom_output,
                    pooling_options);
            });

        const double cudnn_ms = kl::test::benchmarkGpu(
            kl::Device::cuda(),
            options::warmup_iterations,
            options::measured_iterations,
            kl::test::options::benchmark::print_each_iteration,
            "cuDNN",
            [&]()
            {
                kl::test::cudnnAvgPool2dForwardFloat32(
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
            benchmark_name,
            "Kerneloom CUDA",
            kerneloom_ms,
            "cuDNN",
            cudnn_ms);
    }

}

TEST(AvgPool2dForwardCudaFloat32, MatchesCuDNN_RegularLarge)
{
    runCorrectness(
        options::regular_large::batch_size,
        options::regular_large::channels,
        options::regular_large::input_height,
        options::regular_large::input_width,
        options::regular_large::output_height,
        options::regular_large::output_width,
        options::regular_large::kernel_height,
        options::regular_large::kernel_width,
        options::regular_large::stride_height,
        options::regular_large::stride_width,
        options::regular_large::padding_height,
        options::regular_large::padding_width);
}

TEST(AvgPool2dForwardCudaFloat32, MatchesCuDNN_DemandingOddShape)
{
    runCorrectness(
        options::demanding_odd_shape::batch_size,
        options::demanding_odd_shape::channels,
        options::demanding_odd_shape::input_height,
        options::demanding_odd_shape::input_width,
        options::demanding_odd_shape::output_height,
        options::demanding_odd_shape::output_width,
        options::demanding_odd_shape::kernel_height,
        options::demanding_odd_shape::kernel_width,
        options::demanding_odd_shape::stride_height,
        options::demanding_odd_shape::stride_width,
        options::demanding_odd_shape::padding_height,
        options::demanding_odd_shape::padding_width);
}

TEST(AvgPool2dForwardCudaFloat32, MatchesCuDNN_PaddedStrideOne)
{
    runCorrectness(
        options::padded_stride_one::batch_size,
        options::padded_stride_one::channels,
        options::padded_stride_one::input_height,
        options::padded_stride_one::input_width,
        options::padded_stride_one::output_height,
        options::padded_stride_one::output_width,
        options::padded_stride_one::kernel_height,
        options::padded_stride_one::kernel_width,
        options::padded_stride_one::stride_height,
        options::padded_stride_one::stride_width,
        options::padded_stride_one::padding_height,
        options::padded_stride_one::padding_width);
}

TEST(AvgPool2dForwardCudaFloat32, MatchesCuDNN_PaddedStrideTwo)
{
    runCorrectness(
        options::padded_stride_two::batch_size,
        options::padded_stride_two::channels,
        options::padded_stride_two::input_height,
        options::padded_stride_two::input_width,
        options::padded_stride_two::output_height,
        options::padded_stride_two::output_width,
        options::padded_stride_two::kernel_height,
        options::padded_stride_two::kernel_width,
        options::padded_stride_two::stride_height,
        options::padded_stride_two::stride_width,
        options::padded_stride_two::padding_height,
        options::padded_stride_two::padding_width);
}

TEST(AvgPool2dForwardCudaFloat32, BenchmarkAgainstCuDNN_RegularLarge)
{
    runBenchmark(
        "AvgPool2d Forward CUDA Float32 Regular Large",
        options::regular_large::batch_size,
        options::regular_large::channels,
        options::regular_large::input_height,
        options::regular_large::input_width,
        options::regular_large::output_height,
        options::regular_large::output_width,
        options::regular_large::kernel_height,
        options::regular_large::kernel_width,
        options::regular_large::stride_height,
        options::regular_large::stride_width,
        options::regular_large::padding_height,
        options::regular_large::padding_width);
}

TEST(AvgPool2dForwardCudaFloat32, BenchmarkAgainstCuDNN_DemandingOddShape)
{
    runBenchmark(
        "AvgPool2d Forward CUDA Float32 Demanding Odd Shape",
        options::demanding_odd_shape::batch_size,
        options::demanding_odd_shape::channels,
        options::demanding_odd_shape::input_height,
        options::demanding_odd_shape::input_width,
        options::demanding_odd_shape::output_height,
        options::demanding_odd_shape::output_width,
        options::demanding_odd_shape::kernel_height,
        options::demanding_odd_shape::kernel_width,
        options::demanding_odd_shape::stride_height,
        options::demanding_odd_shape::stride_width,
        options::demanding_odd_shape::padding_height,
        options::demanding_odd_shape::padding_width);
}

TEST(AvgPool2dForwardCudaFloat32, BenchmarkAgainstCuDNN_PaddedStrideOne)
{
    runBenchmark(
        "AvgPool2d Forward CUDA Float32 Padded Stride One",
        options::padded_stride_one::batch_size,
        options::padded_stride_one::channels,
        options::padded_stride_one::input_height,
        options::padded_stride_one::input_width,
        options::padded_stride_one::output_height,
        options::padded_stride_one::output_width,
        options::padded_stride_one::kernel_height,
        options::padded_stride_one::kernel_width,
        options::padded_stride_one::stride_height,
        options::padded_stride_one::stride_width,
        options::padded_stride_one::padding_height,
        options::padded_stride_one::padding_width);
}

TEST(AvgPool2dForwardCudaFloat32, BenchmarkAgainstCuDNN_PaddedStrideTwo)
{
    runBenchmark(
        "AvgPool2d Forward CUDA Float32 Padded Stride Two",
        options::padded_stride_two::batch_size,
        options::padded_stride_two::channels,
        options::padded_stride_two::input_height,
        options::padded_stride_two::input_width,
        options::padded_stride_two::output_height,
        options::padded_stride_two::output_width,
        options::padded_stride_two::kernel_height,
        options::padded_stride_two::kernel_width,
        options::padded_stride_two::stride_height,
        options::padded_stride_two::stride_width,
        options::padded_stride_two::padding_height,
        options::padded_stride_two::padding_width);
}

#endif // KL_ENABLE_CUDA