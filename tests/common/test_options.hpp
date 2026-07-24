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

    namespace tensor_compare
    {

        constexpr double max_mismatch_ratio =
            1.0e-7;

    }

    namespace linear_forward_float32
    {

        namespace large_non_square
        {

            constexpr std::size_t batch_size =
                5632;

            constexpr std::size_t input_features =
                22528;

            constexpr std::size_t output_features =
                11264;

        }

        namespace demanding_odd_shape
        {

            constexpr std::size_t batch_size =
                5633;

            constexpr std::size_t input_features =
                22527;

            constexpr std::size_t output_features =
                11265;

        }

        namespace bias_add_large
        {

            constexpr std::size_t batch_size =
                28672;

            constexpr std::size_t input_features =
                512;

            constexpr std::size_t output_features =
                32768;

        }

        namespace bias_add_odd
        {

            constexpr std::size_t batch_size =
                28673;

            constexpr std::size_t input_features =
                513;

            constexpr std::size_t output_features =
                32769;

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

    namespace linear_backward_float32
    {

        namespace large_non_square
        {

            constexpr std::size_t batch_size =
                8192;

            constexpr std::size_t input_features =
                17408;

            constexpr std::size_t output_features =
                8704;

        }

        namespace demanding_odd_shape
        {

            constexpr std::size_t batch_size =
                8193;

            constexpr std::size_t input_features =
                17407;

            constexpr std::size_t output_features =
                8705;

        }

        namespace grad_bias_large
        {

            constexpr std::size_t batch_size =
                32768;

            constexpr std::size_t output_features =
                65536;

        }

        namespace grad_bias_odd
        {

            constexpr std::size_t batch_size =
                32769;

            constexpr std::size_t output_features =
                65537;

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

    namespace avgpool2d_forward_float32
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

    namespace activation_forward_float32
    {

        namespace elementwise_huge
        {

            constexpr std::size_t element_count =
                64ULL * 1024ULL * 1024ULL;

        }

        namespace elementwise_odd_huge
        {

            constexpr std::size_t element_count =
                67ULL * 1024ULL * 1024ULL - 1ULL;

        }

        namespace softmax_huge
        {

            constexpr std::size_t batch_size =
                32768;

            constexpr std::size_t class_count =
                1024;

        }

        namespace softmax_odd_huge
        {

            constexpr std::size_t batch_size =
                65537;

            constexpr std::size_t class_count =
                1009;

        }

        constexpr double elementwise_absolute_tolerance =
            1.0e-5;

        constexpr double elementwise_relative_tolerance =
            1.0e-5;

        constexpr double softmax_absolute_tolerance =
            1.0e-4;

        constexpr double softmax_relative_tolerance =
            1.0e-4;

        constexpr std::size_t warmup_iterations =
            3;

        constexpr std::size_t measured_iterations =
            10;

    }

    namespace activation_backward_float32
    {

        namespace elementwise_huge
        {

            constexpr std::size_t element_count =
                64ULL * 1024ULL * 1024ULL;

        }

        namespace elementwise_odd_huge
        {

            constexpr std::size_t element_count =
                67ULL * 1024ULL * 1024ULL - 1ULL;

        }

        namespace softmax_huge
        {

            constexpr std::size_t batch_size =
                32768;

            constexpr std::size_t class_count =
                1024;

        }

        namespace softmax_odd_huge
        {

            constexpr std::size_t batch_size =
                65537;

            constexpr std::size_t class_count =
                1009;

        }

        constexpr double elementwise_absolute_tolerance =
            1.0e-4;

        constexpr double elementwise_relative_tolerance =
            1.0e-4;

        constexpr double softmax_absolute_tolerance =
            1.0e-4;

        constexpr double softmax_relative_tolerance =
            1.0e-4;

        constexpr std::size_t warmup_iterations =
            3;

        constexpr std::size_t measured_iterations =
            10;

    }

}

#endif // KL_TEST_OPTIONS_HPP