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

}

#endif // KL_TEST_OPTIONS_HPP