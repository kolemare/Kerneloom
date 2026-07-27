#ifndef KL_TEST_LINEAR_BACKWARD_OPTIONS_HPP
#define KL_TEST_LINEAR_BACKWARD_OPTIONS_HPP

#include <array>
#include <cstddef>

namespace kl::test::options::linear_backward
{

    struct Shape
    {
        const char *name;
        std::size_t batch_size;
        std::size_t input_features;
        std::size_t output_features;
    };

    struct BiasShape
    {
        const char *name;
        std::size_t batch_size;
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

    inline constexpr std::array<BiasShape, 8> grad_bias_correctness_shapes =
        {
            BiasShape{
                "Single Element",
                1,
                1,
            },
            BiasShape{
                "Single Batch Small",
                1,
                5,
            },
            BiasShape{
                "Single Output Feature",
                8,
                1,
            },
            BiasShape{
                "Small Odd",
                7,
                11,
            },
            BiasShape{
                "Medium Odd",
                33,
                129,
            },
            BiasShape{
                "Tall Small",
                1024,
                16,
            },
            BiasShape{
                "Wide Output Small",
                16,
                1024,
            },
            BiasShape{
                "Batch One Large Output",
                1,
                4096,
            },
    };

    inline constexpr std::array<Shape, 11> benchmark_shapes =
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
                8192,
                17408,
                8704,
            },
            Shape{
                "Stress Demanding Odd Shape",
                8193,
                17407,
                8705,
            },
    };

    inline constexpr std::array<BiasShape, 5> grad_bias_benchmark_shapes =
        {
            BiasShape{
                "Bias Small Realistic",
                32,
                512,
            },
            BiasShape{
                "Bias Transformer Projection",
                256,
                3072,
            },
            BiasShape{
                "Bias Tall Batch",
                4096,
                4096,
            },
            BiasShape{
                "Bias Add Large",
                28672,
                65536,
            },
            BiasShape{
                "Bias Add Odd",
                28673,
                65537,
            },
    };

    inline constexpr double float32_absolute_tolerance =
        1.0e-3;

    inline constexpr double float32_relative_tolerance =
        1.0e-3;

    inline constexpr std::size_t warmup_iterations =
        3;

    inline constexpr std::size_t measured_iterations =
        5;

}

#endif // KL_TEST_LINEAR_BACKWARD_OPTIONS_HPP