#ifndef KL_TEST_ACTIVATION_FORWARD_OPTIONS_HPP
#define KL_TEST_ACTIVATION_FORWARD_OPTIONS_HPP

#include <array>
#include <cstddef>

namespace kl::test::options::activation_forward
{

    struct ElementwiseShape
    {
        const char *name;
        std::size_t element_count;
    };

    struct SoftmaxShape
    {
        const char *name;
        std::size_t batch_size;
        std::size_t class_count;
    };

    inline constexpr std::array<ElementwiseShape, 7> elementwise_correctness_shapes =
        {
            ElementwiseShape{
                "Single Element",
                1,
            },
            ElementwiseShape{
                "Tiny Odd",
                17,
            },
            ElementwiseShape{
                "One Warp",
                32,
            },
            ElementwiseShape{
                "One Block",
                256,
            },
            ElementwiseShape{
                "Odd Medium",
                65537,
            },
            ElementwiseShape{
                "One Meg Elements",
                1024ULL * 1024ULL,
            },
            ElementwiseShape{
                "Huge",
                256ULL * 1024ULL * 1024ULL,
            },
    };

    inline constexpr std::array<ElementwiseShape, 6> elementwise_benchmark_shapes =
        {
            ElementwiseShape{
                "Tiny Launch Overhead",
                1024,
            },
            ElementwiseShape{
                "Small Realistic",
                32ULL * 64ULL * 112ULL * 112ULL,
            },
            ElementwiseShape{
                "Medium Realistic",
                64ULL * 128ULL * 56ULL * 56ULL,
            },
            ElementwiseShape{
                "Large Realistic",
                128ULL * 256ULL * 28ULL * 28ULL,
            },
            ElementwiseShape{
                "Huge",
                256ULL * 1024ULL * 1024ULL,
            },
            ElementwiseShape{
                "Huge Odd",
                268435456ULL - 1ULL,
            },
    };

    inline constexpr std::array<SoftmaxShape, 8> softmax_correctness_shapes =
        {
            SoftmaxShape{
                "Single Row Tiny",
                1,
                2,
            },
            SoftmaxShape{
                "Single Row Odd",
                1,
                17,
            },
            SoftmaxShape{
                "Small Batch Small Classes",
                8,
                10,
            },
            SoftmaxShape{
                "Medium Batch Classifier",
                256,
                1000,
            },
            SoftmaxShape{
                "Transformer Tokens",
                4096,
                768,
            },
            SoftmaxShape{
                "Large Classifier",
                8192,
                2048,
            },
            SoftmaxShape{
                "Odd Shape",
                8193,
                1009,
            },
            SoftmaxShape{
                "Huge",
                131072,
                2048,
            },
    };

    inline constexpr std::array<SoftmaxShape, 6> softmax_benchmark_shapes =
        {
            SoftmaxShape{
                "Tiny Launch Overhead",
                1,
                128,
            },
            SoftmaxShape{
                "Small Classifier",
                1024,
                1000,
            },
            SoftmaxShape{
                "Transformer Tokens",
                4096,
                768,
            },
            SoftmaxShape{
                "Large Classifier",
                8192,
                2048,
            },
            SoftmaxShape{
                "Huge",
                131072,
                2048,
            },
            SoftmaxShape{
                "Huge Odd",
                262145,
                1009,
            },
    };

    inline constexpr double elementwise_float32_absolute_tolerance =
        1.0e-5;

    inline constexpr double elementwise_float32_relative_tolerance =
        1.0e-5;

    inline constexpr double softmax_float32_absolute_tolerance =
        1.0e-4;

    inline constexpr double softmax_float32_relative_tolerance =
        1.0e-4;

    inline constexpr std::size_t warmup_iterations =
        3;

    inline constexpr std::size_t measured_iterations =
        10;

}

#endif // KL_TEST_ACTIVATION_FORWARD_OPTIONS_HPP