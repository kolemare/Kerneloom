#ifndef KL_TEST_LINEAR_FORWARD_OPTIONS_HPP
#define KL_TEST_LINEAR_FORWARD_OPTIONS_HPP

#include <array>
#include <cstddef>

namespace kl::test::options::linear_forward
{

    struct Shape
    {
        const char *name;
        std::size_t batch_size;
        std::size_t input_features;
        std::size_t output_features;
    };

    inline constexpr std::array<Shape, 10> correctness_shapes =
        {
            Shape{
                "Single Element",
                1,
                1,
                1,
            },
            Shape{
                "Single Batch Small",
                1,
                7,
                5,
            },
            Shape{
                "Single Input Feature",
                8,
                1,
                13,
            },
            Shape{
                "Single Output Feature",
                8,
                13,
                1,
            },
            Shape{
                "Small Square",
                16,
                16,
                16,
            },
            Shape{
                "Small Odd",
                7,
                13,
                11,
            },
            Shape{
                "Medium Odd",
                33,
                257,
                129,
            },
            Shape{
                "Wide Small",
                16,
                1024,
                17,
            },
            Shape{
                "Tall Small",
                1024,
                17,
                16,
            },
            Shape{
                "Batch One Large Features",
                1,
                4096,
                4096,
            },
    };

    inline constexpr std::array<Shape, 11> no_bias_benchmark_shapes =
        {
            Shape{
                "Tiny Launch Overhead",
                1,
                128,
                128,
            },
            Shape{
                "Small Realistic MLP",
                32,
                512,
                512,
            },
            Shape{
                "Medium Realistic MLP",
                128,
                1024,
                1024,
            },
            Shape{
                "Transformer Projection",
                256,
                768,
                3072,
            },
            Shape{
                "Transformer Feed Forward",
                128,
                4096,
                11008,
            },
            Shape{
                "Tall Batch Moderate Features",
                2048,
                512,
                512,
            },
            Shape{
                "Skinny Input Wide Output",
                1024,
                128,
                4096,
            },
            Shape{
                "Wide Input Skinny Output",
                1024,
                4096,
                128,
            },
            Shape{
                "Large Square",
                4096,
                4096,
                4096,
            },
            Shape{
                "Stress Large Non-Square",
                5632,
                22528,
                11264,
            },
            Shape{
                "Stress Demanding Odd Shape",
                5633,
                22527,
                11265,
            },
    };

    inline constexpr std::array<Shape, 5> bias_benchmark_shapes =
        {
            Shape{
                "Bias Small Realistic",
                32,
                512,
                512,
            },
            Shape{
                "Bias Transformer Projection",
                256,
                768,
                3072,
            },
            Shape{
                "Bias Tall Batch",
                4096,
                512,
                4096,
            },
            Shape{
                "Bias Add Large",
                28672,
                512,
                32768,
            },
            Shape{
                "Bias Add Odd",
                28673,
                513,
                32769,
            },
    };

    inline constexpr double float32_absolute_tolerance =
        1.0e-3;

    inline constexpr double float32_relative_tolerance =
        1.0e-3;

    inline constexpr std::size_t warmup_iterations =
        3;

    inline constexpr std::size_t measured_iterations =
        10;

}

#endif // KL_TEST_LINEAR_FORWARD_OPTIONS_HPP