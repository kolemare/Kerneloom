#ifndef KL_TEST_OPTIONS_HPP
#define KL_TEST_OPTIONS_HPP

#include <cstddef>

namespace kl::test::options
{

    namespace benchmark
    {

        constexpr bool print_each_iteration =
            false;

    }

    namespace linear_forward_float32
    {

        namespace large_non_square
        {

            constexpr std::size_t batch_size =
                2048;

            constexpr std::size_t input_features =
                8192;

            constexpr std::size_t output_features =
                4096;

        }

        namespace demanding_odd_shape
        {

            constexpr std::size_t batch_size =
                2049;

            constexpr std::size_t input_features =
                8191;

            constexpr std::size_t output_features =
                4097;

        }

        constexpr double absolute_tolerance =
            1.0e-3;

        constexpr double relative_tolerance =
            1.0e-3;

        constexpr std::size_t warmup_iterations =
            3;

        constexpr std::size_t measured_iterations =
            5;

    }

    namespace maxpool2d_forward_float32
    {

        namespace regular_large
        {

            constexpr std::size_t batch_size =
                448;

            constexpr std::size_t channels =
                64;

            constexpr std::size_t input_height =
                256;

            constexpr std::size_t input_width =
                256;

            constexpr std::size_t kernel_height =
                2;

            constexpr std::size_t kernel_width =
                2;

            constexpr std::size_t stride_height =
                2;

            constexpr std::size_t stride_width =
                2;

            constexpr std::size_t padding_height =
                0;

            constexpr std::size_t padding_width =
                0;

            constexpr std::size_t output_height =
                (input_height + 2 * padding_height - kernel_height) /
                    stride_height +
                1;

            constexpr std::size_t output_width =
                (input_width + 2 * padding_width - kernel_width) /
                    stride_width +
                1;

        }

        namespace demanding_odd_shape
        {

            constexpr std::size_t batch_size =
                257;

            constexpr std::size_t channels =
                65;

            constexpr std::size_t input_height =
                255;

            constexpr std::size_t input_width =
                257;

            constexpr std::size_t kernel_height =
                2;

            constexpr std::size_t kernel_width =
                2;

            constexpr std::size_t stride_height =
                2;

            constexpr std::size_t stride_width =
                2;

            constexpr std::size_t padding_height =
                0;

            constexpr std::size_t padding_width =
                0;

            constexpr std::size_t output_height =
                (input_height + 2 * padding_height - kernel_height) /
                    stride_height +
                1;

            constexpr std::size_t output_width =
                (input_width + 2 * padding_width - kernel_width) /
                    stride_width +
                1;

        }

        namespace padded_stride_one
        {

            constexpr std::size_t batch_size =
                128;

            constexpr std::size_t channels =
                64;

            constexpr std::size_t input_height =
                224;

            constexpr std::size_t input_width =
                224;

            constexpr std::size_t kernel_height =
                3;

            constexpr std::size_t kernel_width =
                3;

            constexpr std::size_t stride_height =
                1;

            constexpr std::size_t stride_width =
                1;

            constexpr std::size_t padding_height =
                1;

            constexpr std::size_t padding_width =
                1;

            constexpr std::size_t output_height =
                (input_height + 2 * padding_height - kernel_height) /
                    stride_height +
                1;

            constexpr std::size_t output_width =
                (input_width + 2 * padding_width - kernel_width) /
                    stride_width +
                1;

        }

        namespace padded_stride_two
        {

            constexpr std::size_t batch_size =
                256;

            constexpr std::size_t channels =
                64;

            constexpr std::size_t input_height =
                255;

            constexpr std::size_t input_width =
                255;

            constexpr std::size_t kernel_height =
                3;

            constexpr std::size_t kernel_width =
                3;

            constexpr std::size_t stride_height =
                2;

            constexpr std::size_t stride_width =
                2;

            constexpr std::size_t padding_height =
                1;

            constexpr std::size_t padding_width =
                1;

            constexpr std::size_t output_height =
                (input_height + 2 * padding_height - kernel_height) /
                    stride_height +
                1;

            constexpr std::size_t output_width =
                (input_width + 2 * padding_width - kernel_width) /
                    stride_width +
                1;

        }

        constexpr double absolute_tolerance =
            1.0e-5;

        constexpr double relative_tolerance =
            1.0e-5;

        constexpr std::size_t warmup_iterations =
            3;

        constexpr std::size_t measured_iterations =
            10;

    }

}

#endif // KL_TEST_OPTIONS_HPP