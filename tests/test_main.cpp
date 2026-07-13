#include <gtest/gtest.h>

#include "common/benchmark_result.hpp"
#include "common/benchmark_timer.hpp"

#include <cstddef>
#include <iostream>
#include <vector>

TEST(KerneloomTests, GTestIsWorking)
{
    EXPECT_EQ(1 + 1, 2);
}

TEST(KerneloomBenchmarks, InitialCpuTimingSmokeTest)
{
    constexpr std::size_t size = 1'000'000;
    constexpr std::size_t warmup_iterations = 3;
    constexpr std::size_t measured_iterations = 10;

    std::vector<float> a(size, 1.0F);
    std::vector<float> b(size, 2.0F);
    std::vector<float> c(size, 0.0F);

    for (std::size_t warmup = 0; warmup < warmup_iterations; ++warmup)
    {
        for (std::size_t i = 0; i < size; ++i)
        {
            c[i] = a[i] + b[i];
        }
    }

    std::vector<double> runs_ms;
    runs_ms.reserve(measured_iterations);

    for (std::size_t iteration = 0; iteration < measured_iterations; ++iteration)
    {
        kl::test::BenchmarkTimer timer;
        timer.start();

        for (std::size_t i = 0; i < size; ++i)
        {
            c[i] = a[i] + b[i];
        }

        runs_ms.push_back(timer.stopMilliseconds());
    }

    const kl::test::BenchmarkResult result =
        kl::test::summarizeBenchmarkRuns(runs_ms);

    EXPECT_FLOAT_EQ(c.front(), 3.0F);
    EXPECT_FLOAT_EQ(c.back(), 3.0F);
    EXPECT_EQ(result.iterations, measured_iterations);

    std::cout
        << "\n[Benchmark] CPU vector add smoke test\n"
        << "  Iterations: " << result.iterations << '\n'
        << "  Average:    " << result.average_ms << " ms\n"
        << "  Min:        " << result.min_ms << " ms\n"
        << "  Max:        " << result.max_ms << " ms\n";
}