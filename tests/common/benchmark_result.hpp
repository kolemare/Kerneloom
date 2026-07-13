#ifndef KL_TEST_BENCHMARK_RESULT_HPP
#define KL_TEST_BENCHMARK_RESULT_HPP

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

namespace kl::test
{

    struct BenchmarkResult
    {
        double average_ms = 0.0;
        double min_ms = 0.0;
        double max_ms = 0.0;
        std::size_t iterations = 0;
    };

    inline BenchmarkResult summarizeBenchmarkRuns(const std::vector<double> &runs_ms)
    {
        BenchmarkResult result{};

        if (runs_ms.empty())
        {
            return result;
        }

        const auto [min_it, max_it] = std::minmax_element(
            runs_ms.begin(),
            runs_ms.end());

        const double total = std::accumulate(
            runs_ms.begin(),
            runs_ms.end(),
            0.0);

        result.iterations = runs_ms.size();
        result.average_ms = total / static_cast<double>(runs_ms.size());
        result.min_ms = *min_it;
        result.max_ms = *max_it;

        return result;
    }

}

#endif // KL_TEST_BENCHMARK_RESULT_HPP