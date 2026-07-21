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

#include <kernels/cuda/pooling/maxpool2d_with_indices_cuda_float32.cuh>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace
{

    namespace options =
        kl::test::options::maxpool2d_forward_float32;

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

    bool validate_indices(
        const kl::Tensor &input,
        const kl::Tensor &output,
        const kl::Tensor &indices,
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
        std::size_t padding_width,
        double absolute_tolerance)
    {
        const auto input_cpu =
            input.to(
                kl::Device::cpu());

        const auto output_cpu =
            output.to(
                kl::Device::cpu());

        const auto indices_cpu =
            indices.to(
                kl::Device::cpu());

        const float *input_data =
            static_cast<const float *>(input_cpu.data());

        const float *output_data =
            static_cast<const float *>(output_cpu.data());

        const std::int32_t *indices_data =
            static_cast<const std::int32_t *>(indices_cpu.data());

        const std::size_t input_plane_size =
            input_height * input_width;

        const std::size_t input_channel_size =
            channels * input_plane_size;

        const std::size_t output_total =
            batch_size * channels * output_height * output_width;

        for (std::size_t output_index = 0;
             output_index < output_total;
             ++output_index)
        {
            const std::size_t ow =
                output_index % output_width;

            const std::size_t oh =
                (output_index / output_width) % output_height;

            const std::size_t c =
                (output_index / (output_width * output_height)) % channels;

            const std::size_t n =
                output_index / (output_width * output_height * channels);

            const std::int32_t raw_index =
                indices_data[output_index];

            if (raw_index < 0)
            {
                return false;
            }

            const std::size_t input_index =
                static_cast<std::size_t>(raw_index);

            const std::size_t input_n =
                input_index / input_channel_size;

            const std::size_t input_c =
                (input_index / input_plane_size) % channels;

            const std::size_t input_h =
                (input_index / input_width) % input_height;

            const std::size_t input_w =
                input_index % input_width;

            if (input_n != n || input_c != c)
            {
                return false;
            }

            const long h_start =
                static_cast<long>(oh * stride_height) -
                static_cast<long>(padding_height);

            const long w_start =
                static_cast<long>(ow * stride_width) -
                static_cast<long>(padding_width);

            const long h_end =
                h_start + static_cast<long>(kernel_height);

            const long w_end =
                w_start + static_cast<long>(kernel_width);

            if (static_cast<long>(input_h) < h_start ||
                static_cast<long>(input_h) >= h_end ||
                static_cast<long>(input_w) < w_start ||
                static_cast<long>(input_w) >= w_end)
            {
                return false;
            }

            float expected_max =
                -INFINITY;

            for (long h = h_start; h < h_end; ++h)
            {
                if (h < 0 || h >= static_cast<long>(input_height))
                {
                    continue;
                }

                for (long w = w_start; w < w_end; ++w)
                {
                    if (w < 0 || w >= static_cast<long>(input_width))
                    {
                        continue;
                    }

                    const std::size_t candidate_index =
                        ((n * channels + c) * input_height +
                         static_cast<std::size_t>(h)) *
                            input_width +
                        static_cast<std::size_t>(w);

                    const float value =
                        input_data[candidate_index];

                    if (value > expected_max)
                    {
                        expected_max =
                            value;
                    }
                }
            }

            const float indexed_value =
                input_data[input_index];

            const float output_value =
                output_data[output_index];

            if (std::fabs(indexed_value - output_value) >
                static_cast<float>(absolute_tolerance))
            {
                return false;
            }

            if (std::fabs(expected_max - output_value) >
                static_cast<float>(absolute_tolerance))
            {
                return false;
            }
        }

        return true;
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

        kl::Tensor indices(
            kl::Shape{
                batch_size,
                channels,
                output_height,
                output_width},
            kl::DType::Int32,
            kl::Device::cuda());

        const kl::Pooling2dOptions pooling_options =
            make_pooling_options(
                kernel_height,
                kernel_width,
                stride_height,
                stride_width,
                padding_height,
                padding_width);

        kl::maxpool2d_with_indices_cuda_float32(
            input,
            actual,
            indices,
            pooling_options);

        const auto expected =
            kl::test::cudnnMaxPool2dForwardFloat32(
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

        EXPECT_TRUE(validate_indices(
            input,
            actual,
            indices,
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_height,
            kernel_width,
            stride_height,
            stride_width,
            padding_height,
            padding_width,
            options::absolute_tolerance));
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

        kl::Tensor kerneloom_indices(
            kl::Shape{
                batch_size,
                channels,
                output_height,
                output_width},
            kl::DType::Int32,
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

        kl::test::CudnnPoolingDescriptor cudnn_pooling_descriptor(
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
            "Kerneloom CUDA With Indices",
            [&]()
            {
                kl::maxpool2d_with_indices_cuda_float32(
                    input,
                    kerneloom_output,
                    kerneloom_indices,
                    pooling_options);
            });

        const double cudnn_ms = kl::test::benchmarkGpu(
            kl::Device::cuda(),
            options::warmup_iterations,
            options::measured_iterations,
            kl::test::options::benchmark::print_each_iteration,
            "cuDNN Value Only",
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

        EXPECT_TRUE(validate_indices(
            input,
            kerneloom_output,
            kerneloom_indices,
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_height,
            kernel_width,
            stride_height,
            stride_width,
            padding_height,
            padding_width,
            options::absolute_tolerance));

        kl::test::printBenchmarkComparison(
            benchmark_name,
            "Kerneloom CUDA With Indices",
            kerneloom_ms,
            "cuDNN Value Only",
            cudnn_ms);
    }

}

TEST(MaxPool2dWithIndicesForwardCudaFloat32, MatchesCuDNN_RegularLarge)
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

TEST(MaxPool2dWithIndicesForwardCudaFloat32, MatchesCuDNN_DemandingOddShape)
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

TEST(MaxPool2dWithIndicesForwardCudaFloat32, MatchesCuDNN_PaddedStrideOne)
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

TEST(MaxPool2dWithIndicesForwardCudaFloat32, MatchesCuDNN_PaddedStrideTwo)
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

TEST(MaxPool2dWithIndicesForwardCudaFloat32, BenchmarkAgainstCuDNN_RegularLarge)
{
    runBenchmark(
        "MaxPool2d With Indices Forward CUDA Float32 Regular Large",
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

TEST(MaxPool2dWithIndicesForwardCudaFloat32, BenchmarkAgainstCuDNN_DemandingOddShape)
{
    runBenchmark(
        "MaxPool2d With Indices Forward CUDA Float32 Demanding Odd Shape",
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

TEST(MaxPool2dWithIndicesForwardCudaFloat32, BenchmarkAgainstCuDNN_PaddedStrideOne)
{
    runBenchmark(
        "MaxPool2d With Indices Forward CUDA Float32 Padded Stride One",
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

TEST(MaxPool2dWithIndicesForwardCudaFloat32, BenchmarkAgainstCuDNN_PaddedStrideTwo)
{
    runBenchmark(
        "MaxPool2d With Indices Forward CUDA Float32 Padded Stride Two",
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