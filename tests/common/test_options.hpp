#ifndef KL_TEST_OPTIONS_HPP
#define KL_TEST_OPTIONS_HPP

#include <cstddef>

namespace kl::test::options
{

    namespace linear_forward_float32
    {

        constexpr std::size_t batch_size =
            2048;

        constexpr std::size_t input_features =
            8192;

        constexpr std::size_t output_features =
            4096;

        constexpr double absolute_tolerance =
            1.0e-3;

        constexpr double relative_tolerance =
            1.0e-3;

        constexpr std::size_t warmup_iterations =
            3;

        constexpr std::size_t measured_iterations =
            5;

        constexpr bool print_each_iteration =
            true;

    }

    namespace maxpool2d_forward_float32
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

        constexpr double absolute_tolerance =
            1.0e-5;

        constexpr double relative_tolerance =
            1.0e-5;

        constexpr std::size_t warmup_iterations =
            3;

        constexpr std::size_t measured_iterations =
            10;

        constexpr bool print_each_iteration =
            true;

    }
}

#endif // KL_TEST_OPTIONS_HPP