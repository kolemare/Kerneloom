#ifndef KL_TEST_BENCHMARK_REPORT_HPP
#define KL_TEST_BENCHMARK_REPORT_HPP

#include <iostream>

namespace kl::test
{

    inline void printBenchmarkComparison(
        const char *name,
        const char *kerneloom_name,
        double kerneloom_ms,
        const char *vendor_name,
        double vendor_ms)
    {
        std::cout
            << "\n[Benchmark] " << name << '\n'
            << "  " << kerneloom_name << ": " << kerneloom_ms << " ms\n"
            << "  " << vendor_name << ": " << vendor_ms << " ms\n";

        if (vendor_ms > 0.0)
        {
            std::cout
                << "  Ratio: " << kerneloom_ms / vendor_ms << "x\n";
        }
    }

}

#endif // KL_TEST_BENCHMARK_REPORT_HPP